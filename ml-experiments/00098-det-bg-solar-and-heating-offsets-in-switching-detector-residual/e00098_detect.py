"""Detectors D0, D1, and D2 on the stored residual, and injected-transfer scoring.

D0 and D1 are experiment 43's. D2 is D0 with the neighbour veto: every D0 alarm stands unless at
least two neighbours stepped the same way and none stepped the opposite way. D2's threshold is
matched on odd months so that its unexplained-alarm count equals D0's.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from typing import Final, Literal

import holidays
import numpy as np
import polars as pl
from e00098_common import (
    EXPERIMENT_43_DIR,
    GRID_LENGTH,
    GRID_START,
    LABELLED_PATH,
    OPERATING_POINTS,
    SERIES_IDS,
    TUNING_MONTHS,
    load_panel,
    near_log_masks,
    stamp_months,
    stamp_of,
)
from e00098_detectors import DetectorInputs, alarms, neighbour_check, prepare
from e00098_inject import QUIET_STAMPS, Event, batches, draw_events, events_frame, inject_power
from e00098_intervals import Interval, build_intervals, relative_residual
from e00098_score import ONSET_TOLERANCE, by_series, end_tolerance, match_event

SEED_43: Final[int] = 43
"""Experiment 43's seed: batch `i` of its 600 events is injected with `SEED_43 + i`."""
SEED_NEW: Final[int] = 5500
"""Seed of the larger sets; batch `i` is injected with `SEED_NEW + i`."""
EXTRA_HEADLINE: Final[int] = 1000
HOLIDAY_EVENTS: Final[int] = 300
HOLIDAY_YEARS: Final[range] = range(2021, 2028)
SETTINGS: Final[dict[str, tuple[Literal["D0", "D1", "D0M", "D2"], int]]] = {
    "D0_8": ("D0", 8),
    "D1_8": ("D1", 8),
    "D0_48": ("D0", 48),
    "D1_48": ("D1", 48),
    "D0M_8": ("D0M", 8),
    "D0M_48": ("D0M", 48),
    "D2_8": ("D2", 8),
    "D2_48": ("D2", 48),
}
"""Detector and window in half-hours (4 h and 24 h) of every setting that is scored."""
OLD_THRESHOLDS: Final[dict[str, float]] = {
    "D0_8": 6.0,
    "D1_8": 3.75,
    "D0_48": 4.0,
    "D1_48": 2.5,
}
"""Experiment 43's thresholds, in units of each series' odd-month noise scale."""
TARGET_COUNTS: Final[dict[int, int]] = {8: 120, 48: 80}
"""D0's odd-month unexplained-alarm counts at experiment 43's thresholds, by window."""
OLD_D0_THRESHOLD: Final[dict[int, float]] = {8: 6.0, 48: 4.0}
NO_VETO: Final[float] = float("inf")
"""`D0M` is D0 whose threshold is matched by the same search as D2's, with the veto switched off."""
VETO_ZS: Final[tuple[float, ...]] = (1.0, 1.25, 1.5)
DEFAULT_VETO_Z: Final[float] = 2.0
GRID_STEP: Final[float] = 0.05
COARSE_STEP: Final[float] = 0.25
SCAN_FLOOR: Final[float] = 1.0
EXPLAIN_PAD: Final[int] = 48
"""An alarm within 24 hours of a logged closure on its series is explained by that closure."""
CHRISTMAS_NAMES: Final[tuple[str, ...]] = ("Christmas", "Boxing", "New Year")
HOLIDAY_NEAR_STAMPS: Final[int] = 48
"""A holiday-set transfer starts within 1 day of a bank holiday outside Christmas and New Year."""
N_WORKERS: Final[int] = 4

Arrays = dict[int, np.ndarray]
RawResult = dict[str, tuple[pl.DataFrame, list[Interval]]]


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


def explained_flags(*, table: pl.DataFrame, logged: dict[int, np.ndarray]) -> np.ndarray:
    """Whether each alarm lies within 24 hours of a logged closure on its series."""
    if table.is_empty():
        return np.zeros(0, dtype=bool)
    return np.array([bool(logged[s][t]) for s, t in table.select("series", "stamp").iter_rows()])


def odd_unexplained(*, table: pl.DataFrame, logged: dict[int, np.ndarray]) -> int:
    """Count the alarms in odd months that no logged closure explains."""
    if table.is_empty():
        return 0
    odd = np.isin(stamp_months()[table["stamp"].to_numpy()], list(TUNING_MONTHS))
    return int((odd & ~explained_flags(table=table, logged=logged)).sum())


def kept_alarms(*, inputs: DetectorInputs, threshold: float, veto_z: float) -> pl.DataFrame:
    """D2's alarms at one threshold: D0's alarms with the neighbour annotation, vetoed included."""
    table = alarms(inputs=inputs, detector="D0", threshold=threshold)
    if table.is_empty():
        return table.with_columns(
            n_neighbours=pl.lit(0, pl.Int64),
            n_opposite=pl.lit(0, pl.Int64),
            n_same=pl.lit(0, pl.Int64),
            tier=pl.lit("", pl.String),
        )
    return neighbour_check(inputs=inputs, table=table, veto_z=veto_z)


def tune_d2(
    *,
    inputs: DetectorInputs,
    veto_z: float,
    logged: dict[int, np.ndarray],
    target: int | None = None,
) -> tuple[float, list[tuple[float, int]]]:
    """Return the lowest threshold on a 0.05 grid whose odd-month unexplained count is at most D0's.

    The count is of D2's unexplained alarms after the veto. The scan moves down from D0's threshold
    plus 1.5 in steps of 0.25 to the first count above the target, then up in steps of 0.05, then
    down while the count stays at or under the target. The second return value holds every
    (threshold, count) pair evaluated. `target` overrides D0's count for the window.
    """
    target = TARGET_COUNTS[inputs.window] if target is None else target
    old = OLD_D0_THRESHOLD[inputs.window]
    curve: dict[float, int] = {}

    def count(threshold: float) -> int:
        key = round(threshold, 2)
        if key not in curve:
            table = kept_alarms(inputs=inputs, threshold=key, veto_z=veto_z)
            curve[key] = odd_unexplained(
                table=table.filter(pl.col("tier") != "vetoed"), logged=logged
            )
        return curve[key]

    threshold = round(old + 1.5, 2)
    while threshold > SCAN_FLOOR and count(threshold) <= target:
        threshold = round(threshold - COARSE_STEP, 2)
    threshold = max(threshold, SCAN_FLOOR)
    while count(threshold) > target:
        threshold = round(threshold + GRID_STEP, 2)
    while threshold - GRID_STEP >= SCAN_FLOOR and count(threshold - GRID_STEP) <= target:
        threshold = round(threshold - GRID_STEP, 2)
    return threshold, sorted(curve.items())


def detect_all(
    *,
    residual: Arrays,
    expected: Arrays,
    scales: dict[int, dict[int, float]],
    thresholds: dict[str, float],
    veto_z: float,
) -> RawResult:
    """Alarms and plain-variant intervals for each setting at the given thresholds.

    `thresholds` has a key for every setting in `SETTINGS`; a D2 setting's table keeps its vetoed
    alarms (tier `vetoed`) but its intervals come from the other alarms only, on the alarm series.
    """
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = {w: prepare(residuals=residual, window=w, scale=scales.get(w)) for w in (8, 48)}
    out: RawResult = {}
    for setting, (detector, window) in SETTINGS.items():
        if detector in ("D2", "D0M"):
            table = kept_alarms(
                inputs=inputs[window],
                threshold=thresholds[setting],
                veto_z=veto_z if detector == "D2" else NO_VETO,
            )
            use = table.filter(pl.col("tier") != "vetoed").with_columns(partners=pl.lit(""))
            intervals = build_intervals(
                alarm_table=use, relative=relative, window=window, detector="D0"
            )
        else:
            table = alarms(inputs=inputs[window], detector=detector, threshold=thresholds[setting])
            intervals = build_intervals(
                alarm_table=table, relative=relative, window=window, detector=detector
            )
        out[setting] = (table, intervals)
    return out


def named_receiver(
    *, event: Event, intervals: dict[int, list[Interval]], table: pl.DataFrame
) -> bool | None:
    """Whether a confirmed alarm that found the event also names another of its metered series.

    None when the event has no metered receiver or no interval found it.
    """
    if not event.receivers:
        return None
    tolerance = end_tolerance(onset=event.onset, end=event.end)
    for series in event.series:
        for interval in intervals.get(series, []):
            if abs(interval.onset - event.onset) > ONSET_TOLERANCE:
                continue
            if abs(interval.end - event.end) > tolerance:
                continue
            row = table.filter(
                pl.col("series") == interval.series, pl.col("stamp") == interval.onset
            )
            if row.is_empty() or row["tier"].item() != "confirmed":
                continue
            partners = {int(p.split(":")[0]) for p in row["partners"].item().split(";") if p}
            if partners & (set(event.series) - {interval.series}):
                return True
    return False


def rows_for_events(*, batch: list[Event], results: RawResult) -> list[dict]:
    """Match every event of a batch against each setting's intervals."""
    rows: list[dict] = []
    for setting, (table, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "setting": setting,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                    "known_delay": match.known_delay,
                    "named": named_receiver(event=event, intervals=grouped, table=table)
                    if match.found and setting.startswith("D2")
                    else None,
                }
            )
    return rows


_STATE: dict = {}


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel, scales, thresholds, and veto strength."""
    _STATE.update(state)


def score_batch(args: tuple[int, list[Event]]) -> list[dict]:
    """Inject one batch with the fast residual, detect, and return one row per event and setting."""
    seed, batch = args
    panel, power = _STATE["panel"], _STATE["panel"]["power"]
    injection = inject_power(power=power, events=batch, seed=seed)
    results = detect_all(
        residual=injection.fast_residual(panel["residual"]),
        expected=panel["expected"],
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
        veto_z=_STATE["veto_z"],
    )
    return rows_for_events(batch=batch, results=results)


def score_events(
    *,
    panel: dict,
    scales: dict[int, dict[int, float]],
    thresholds: dict[str, float],
    veto_z: float,
    groups: list[tuple[int, list[Event]]],
) -> pl.DataFrame:
    """Score batches of events, each with its own injection seed, in forked workers."""
    state = {"panel": panel, "scales": scales, "thresholds": thresholds, "veto_z": veto_z}
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        rows = [r for part in pool.map(score_batch, groups) for r in part]
    return pl.DataFrame(rows, infer_schema_length=None)


def seeded_batches(*, events: list[Event], seed: int) -> list[tuple[int, list[Event]]]:
    """Split events into batches that share no series within 7 days; batch `i` has seed + i."""
    return [(seed + i, batch) for i, batch in enumerate(batches(events=events))]


def load() -> tuple[dict, dict[int, dict[int, float]]]:
    """Return the stored panel and its odd-month noise scales for both windows."""
    panel = load_panel()
    scales = {w: prepare(residuals=panel["residual"], window=w).scale for w in (8, 48)}
    return panel, scales


def real_alarm_rows(*, results: RawResult, logged: dict[int, np.ndarray]) -> pl.DataFrame:
    """One row per real alarm, flagged for logged explanation, month, and holiday distance."""
    months, distance = stamp_months(), holiday_distance_by_stamp()
    pieces = []
    for setting, (table, _) in results.items():
        if table.is_empty():
            continue
        stamps = table["stamp"].to_numpy()
        tier = table["tier"] if "tier" in table.columns else pl.Series([""] * table.height)
        pieces.append(
            table.select("series", "stamp", "step_mw", "z").with_columns(
                setting=pl.lit(setting),
                tier=tier,
                logged_matched=pl.Series(explained_flags(table=table, logged=logged)),
                month=pl.Series(months[stamps]),
                holiday_distance=pl.Series(distance[stamps]),
            )
        )
    return pl.concat(pieces)


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


def score_real(*, results: RawResult, real: pl.DataFrame) -> pl.DataFrame:
    """Match the labelled real transfers against each setting's intervals on the real residual."""
    rows = []
    for setting, (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for row in real.iter_rows(named=True):
            series = tuple(int(s) for s in row["series"].split(","))
            match = match_event(
                series=series, onset=row["onset"], end=row["end"], intervals=grouped
            )
            rows.append(
                {
                    **row,
                    "setting": setting,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def neighbour_check_for_cards(*, cards: pl.DataFrame, veto_z: float) -> pl.DataFrame:
    """Give each labelled alarm card the tier its alarm would receive.

    `cards` has `detector` (an experiment 43 setting name), `stamp`, and `series`. The 24-hour
    cards use the 24-hour neighbour steps and the 4-hour cards use the 4-hour ones.
    """
    panel, scales = load()
    pieces = []
    for window in (8, 48):
        inputs = prepare(residuals=panel["residual"], window=window, scale=scales[window])
        part = cards.filter(pl.col("detector").str.contains(f"_{window}_"))
        table = part.select(
            pl.col("series").cast(pl.Int64), pl.col("stamp").cast(pl.Int64), partners=pl.lit("")
        )
        tiers = neighbour_check(inputs=inputs, table=table, veto_z=veto_z)["tier"]
        pieces.append(part.with_columns(tier=tiers))
    return pl.concat(pieces)
