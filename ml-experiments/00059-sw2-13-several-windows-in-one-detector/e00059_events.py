"""Event sets, quiet masks, and the labelled real transfers, as in experiment 58's harness.

The 1,900 injected transfers are experiment 58's: 600 from experiment 43's draw, 1,000 more in the
headline band, and 300 headline-band transfers within 1 day of a bank holiday.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Final

import holidays
import numpy as np
import polars as pl
from e00059_common import (
    EXPERIMENT_43_DIR,
    GRID_LENGTH,
    GRID_START,
    LABELLED_PATH,
    OPERATING_POINTS,
    SERIES_IDS,
    near_log_masks,
    stamp_of,
)
from e00059_inject import QUIET_STAMPS, Event, batches, draw_events, events_frame

SEED_43: Final[int] = 43
"""Experiment 43's seed: batch `i` of its 600 events is injected with `SEED_43 + i`."""
SEED_NEW: Final[int] = 5500
"""Seed of the larger sets; batch `i` is injected with `SEED_NEW + i`."""
EXTRA_HEADLINE: Final[int] = 1000
HOLIDAY_YEARS: Final[range] = range(2021, 2028)
HOLIDAY_EVENTS: Final[int] = 300
CHRISTMAS_NAMES: Final[tuple[str, ...]] = ("Christmas", "Boxing", "New Year")
EXPLAIN_PAD: Final[int] = 48
"""An alarm within 24 hours of a logged closure on its series is explained by that closure."""

Arrays = dict[int, np.ndarray]


def bank_holidays_outside_christmas() -> np.ndarray:
    """Return the bank holiday dates in England other than Christmas, Boxing Day, and New Year."""
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    kept = [d for d, n in calendar.items() if not any(word in n for word in CHRISTMAS_NAMES)]
    return np.array(sorted(kept), dtype="datetime64[D]")


def stamp_dates() -> np.ndarray:
    """Return the local (Europe/London) calendar date of every grid stamp, as `datetime64[D]`."""
    index = pl.datetime_range(
        GRID_START,
        GRID_START + timedelta(minutes=30 * (GRID_LENGTH - 1)),
        "30m",
        eager=True,
        time_zone="UTC",
    )
    return index.dt.convert_time_zone("Europe/London").dt.date().to_numpy().astype("datetime64[D]")


def holiday_distance_by_stamp() -> np.ndarray:
    """Return, for every grid stamp, the signed days from its local date to the nearest holiday.

    Positive after the holiday. Only bank holidays outside Christmas and New Year count.
    """
    dates = stamp_dates()
    holiday_dates = bank_holidays_outside_christmas()
    unique = np.unique(dates)
    gaps = (unique[:, None] - holiday_dates[None, :]).astype(int)
    nearest = gaps[np.arange(len(unique)), np.abs(gaps).argmin(axis=1)]
    return nearest[np.searchsorted(unique, dates)]


def log_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks within 7 days of a logged closure."""
    return near_log_masks(pad_stamps=QUIET_STAMPS)


def alarm_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks within 2 days of an alarm of experiment 43's four settings."""
    quiet = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for name in OPERATING_POINTS:
        table = pl.read_parquet(EXPERIMENT_43_DIR / f"baseline_alarms_{name}.parquet")
        for series, stamp in table.select("series", "stamp").iter_rows():
            quiet[series][max(stamp - 2 * 48, 0) : stamp + 2 * 48] = True
    return quiet


def quiet_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks that no injected transfer may touch, as in experiment 43.

    No logged closure within 7 days, and no alarm of experiment 43's four settings within 2 days.
    """
    logged, alarmed = log_masks(), alarm_masks()
    return {s: logged[s] | alarmed[s] for s in logged}


def original_events(*, power: Arrays) -> list[Event]:
    """Redraw experiment 43's 600 events and check they equal its stored table."""
    events = draw_events(quiet=quiet_masks(), power=power, seed=SEED_43)
    if not events_frame(events=events).equals(
        pl.read_parquet(EXPERIMENT_43_DIR / "events.parquet")
    ):
        raise ValueError("the redrawn events differ from experiment 43's events.parquet")
    return events


def new_events(*, power: Arrays) -> tuple[list[Event], list[Event]]:
    """Draw the extra headline-band set and the holiday set, with ids after experiment 43's 600."""
    extra = draw_events(
        quiet=quiet_masks(), power=power, seed=SEED_NEW, headline_only=EXTRA_HEADLINE, first_id=1000
    )
    distance = holiday_distance_by_stamp()

    def near_holiday(onset: int) -> bool:
        return bool(abs(distance[onset]) <= 1)

    holiday = draw_events(
        quiet=log_masks(),
        power=power,
        seed=SEED_NEW + 1,
        headline_only=HOLIDAY_EVENTS,
        first_id=3000,
        onset_ok=near_holiday,
    )
    return extra, holiday


def seeded_batches(*, events: list[Event], seed: int) -> list[tuple[int, list[Event]]]:
    """Split events into batches that share no series within 7 days; batch `i` has seed + i."""
    return [(seed + i, batch) for i, batch in enumerate(batches(events=events))]


def _usual_load(*, power: Arrays, series: int, onset: int) -> float:
    """Mean absolute power of a series in the 28 days before `onset`, as in experiment 44."""
    return float(np.nanmean(np.abs(power[series][max(onset - 28 * 48, 0) : onset])))


def real_events(*, power: Arrays) -> pl.DataFrame:
    """The labelled real transfers: stamps, series, visibility, hours, and share of the load."""
    frame = (
        pl.read_parquet(LABELLED_PATH)
        .filter(
            ~pl.col("is_sentinel"),
            pl.col("kind") == "real_operation",
            pl.col("visibility").is_in(["conserved", "one_sided"]),
        )
        .with_columns(
            hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60,
        )
    )
    rows = []
    for row in frame.iter_rows(named=True):
        receivers = [int(r["series"]) for r in json.loads(row["receivers"])]
        series = sorted(
            {
                int(s)
                for s in (row["donor"], row["side_a"], row["side_b"], *receivers)
                if s is not None
            }
        )
        visibility = "one_sided" if row["max_side_missing_share"] > 0.5 else row["visibility"]
        onset = stamp_of(when=row["onset_utc"])
        metered = sorted(set(receivers) & set(power))
        load = sum(_usual_load(power=power, series=r, onset=onset) for r in metered)
        moved = row["moved_mw_p50"]
        share = moved / load if metered and moved is not None and load > 0 else None
        rows.append(
            {
                "card_id": row["card_id"],
                "visibility": visibility,
                "onset": onset,
                "end": stamp_of(when=row["end_utc"]) + 1,
                "hours": row["hours"],
                "share": share,
                "series": ",".join(map(str, series)),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)
