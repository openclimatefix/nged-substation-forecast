"""Event draws, injection, and the stored detector state for experiment 00088.

The draws and the injection are experiment 00082's (`inject_shaped`, with the new shape S6). The
events start in the six even months of the validation year, so every scored transfer starts in a
month that the detector's logistic score and thresholds were not fitted on.
"""

import io
import json
import pickle
from datetime import UTC, datetime
from typing import Final

import e00088_common as common
import e00088_grid as grid
import e00088_labels as labelling
import numpy as np
import polars as pl
from e00088_inject import QUIET_STAMPS, Event, batches, draw_events
from e00088_profile import fit_profile_shares
from e00088_variants import load_both
from scipy.ndimage import maximum_filter1d

SEED_EVENTS: Final[int] = 8800
"""Seed of the event draw; batch `i` of a shape is injected with `SEED_EVENTS + 1000 + i`."""
FIRST_ID: Final[int] = 88_000
PER_MONTH_COUNTS: Final[tuple[int, int]] = (27, 9)
"""Events per scored month: in the headline band, and in each of the other eight bands."""
MIN_PRESENT: Final[float] = 0.9
"""Share of the power that must be present on each series of an event, 7 days either side."""
DRAW_MONTHS: Final[tuple[int, ...]] = tuple(range(1, 13))
"""The calendar months of the validation year (2025-07 to 2026-06) that transfers may start in."""
SCORED_MONTHS: Final[frozenset[int]] = frozenset({2, 4, 6, 8, 10, 12})
"""The even months, which the detector's logistic score and thresholds were not fitted on."""
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
VALIDATION_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
EXPERIMENT_82_DIR = grid.WORK_DIR
OFFSET: Final[int] = int((datetime(2022, 1, 1, tzinfo=UTC) - common.GRID_START) / common.STEP)
"""Half-hours from the dense grid's start (2019-09-01) to experiment 00082's grid (2022-01-01)."""
Arrays = dict[int, np.ndarray]


def validation_stamps() -> tuple[int, int]:
    """Return the first stamp of the validation year and the first stamp after it."""
    return grid.stamp_of(when=VALIDATION_START), grid.stamp_of(when=VALIDATION_END)


def log_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks within 7 days of a logged closure."""
    return grid.near_log_masks(pad_stamps=QUIET_STAMPS)


def alarm_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks within 2 days of an alarm of experiment 43's four settings."""
    quiet = {s: np.zeros(grid.GRID_LENGTH, dtype=bool) for s in grid.SERIES_IDS}
    for name in grid.OPERATING_POINTS:
        table = pl.read_parquet(grid.EXPERIMENT_43_DIR / f"baseline_alarms_{name}.parquet")
        for series, stamp in table.select("series", "stamp").iter_rows():
            quiet[series][max(stamp - 2 * 48, 0) : stamp + 2 * 48] = True
    return quiet


def labelled_closure_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks within 7 days of a closure in experiment 41's labelled set.

    The switching log of experiment 37 ends in March 2026, and the labelled set holds closures to
    June 2026, so this mask keeps April to June from looking quieter than it was.
    """
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    closed = labelling.load_labels(series=series)
    ids = {int(r["time_series_id"]): r["site"] for r in series.iter_rows(named=True)}
    out: dict[int, np.ndarray] = {}
    for sid in grid.SERIES_IDS:
        flags = closed[ids[sid]].closed[OFFSET : OFFSET + grid.GRID_LENGTH]
        out[sid] = maximum_filter1d(flags, size=2 * QUIET_STAMPS + 1)
    return out


def quiet_masks() -> dict[int, np.ndarray]:
    """Series-stamp masks that no injected transfer may touch (as in experiment 43).

    No logged closure within 7 days (the log and the labelled set), and no alarm of experiment 43's
    four settings within 2 days.
    """
    logged, alarmed, labelled = log_masks(), alarm_masks(), labelled_closure_masks()
    return {s: logged[s] | alarmed[s] | labelled[s] for s in logged}


def draw_scored_events(*, power: Arrays) -> list[Event]:
    """Draw events by experiment 82's stratification, equally in each calendar month.

    Each month of the validation year is asked for 27 headline-band events and 9 in each of the
    other eight bands, all starting in that month and quiet on every series they touch. A month
    with no quiet window on any series (August and December 2025) returns few or none.

    Args:
        power: The stored panel's observed power by series.

    Returns:
        The events, numbered from `FIRST_ID`, without those drawn over a data gap.
    """
    first, last = validation_stamps()
    months = grid.stamp_months()
    quiet = quiet_masks()
    drawn: list[Event] = []
    for number, month in enumerate(DRAW_MONTHS):
        drawn += draw_events(
            quiet=quiet,
            power=power,
            seed=SEED_EVENTS + number,
            first_id=FIRST_ID + 1000 * number,
            onset_ok=lambda onset, m=month: first <= onset < last and int(months[onset]) == m,
            counts=PER_MONTH_COUNTS,
        )
    return [event for event in drawn if has_data(event=event, power=power)]


def has_data(*, event: Event, power: Arrays) -> bool:
    """Whether every series of the event has at least 90% of its power from 7 days either side.

    An event drawn over a data gap would inject nothing measurable, so it is left out.
    """
    span = slice(max(event.onset - QUIET_STAMPS, 0), event.end + QUIET_STAMPS)
    return all(float(np.isfinite(power[s][span]).mean()) >= MIN_PRESENT for s in event.series)


def seeded_batches(*, events: list[Event]) -> list[tuple[int, list[Event]]]:
    """Split events into batches sharing no series within 7 days; batch `i` has seed 1000 + i."""
    return [(SEED_EVENTS + 1000 + i, batch) for i, batch in enumerate(batches(events=events))]


def make_state() -> dict:
    """Return the panels, noise scales, fitted shares, fitted score, and threshold of XL.

    The XL logistic score and threshold are experiment 82's, fitted on odd-month injections; nothing
    is refitted here.
    """
    panel, offset_panel = load_both()
    scales = {
        "X": prepare_scales(panel=panel, offset_panel=offset_panel),
    }
    models = _ModelUnpickler(io.BytesIO((EXPERIMENT_82_DIR / "models.pkl").read_bytes())).load()
    thresholds = json.loads((EXPERIMENT_82_DIR / "thresholds.json").read_text())
    return {
        "panel": panel,
        "offset_panel": offset_panel,
        "scales": scales,
        "shares": fit_profile_shares(panel=panel),
        "model": models["XL"],
        "threshold": thresholds["XL"],
    }


class _ModelUnpickler(pickle.Unpickler):
    """Load experiment 82's pickled scores, whose classes now live in this folder's module."""

    def find_class(self, module: str, name: str) -> object:
        """Map experiment 82's `e00082_combined` module to `e00088_combined`."""
        return super().find_class(module.replace("e00082_", "e00088_"), name)


def prepare_scales(*, panel: dict, offset_panel: dict) -> dict[int, float]:
    """Return the odd-month noise scale of every series for the X step statistic."""
    from e00088_variants import scales_for

    return scales_for(variant="X", residual=panel["residual"], offset=offset_panel["residual"])


def cached_events(*, power: Arrays) -> list[Event]:
    """Return the drawn events, drawing and saving them on the first call."""
    path = common.WORK_DIR / "events.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    events = draw_scored_events(power=power)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(events))
    return events
