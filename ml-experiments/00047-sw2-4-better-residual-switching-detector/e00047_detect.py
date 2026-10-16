"""Residual panels per arm, threshold matching on odd months, and scoring of injected transfers.

Everything here follows experiment 43's harness (`e00047_detectors`, `e00047_intervals`,
`e00047_inject`, `e00047_score` are copies of its modules) except that the threshold of each of the
four settings is matched to an unexplained-alarm count on odd months rather than fixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
from e00047_common import (
    EXPERIMENT_43_DIR,
    FEATURES_PATH,
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
from e00047_detectors import DetectorInputs, DetectorType, alarms, prepare
from e00047_fit import HOLIDAY_YEARS
from e00047_fit import bank_holiday_dates as bank_holiday_dates_all
from e00047_inject import QUIET_STAMPS, Event, draw_events, events_frame, inject_power
from e00047_intervals import build_intervals, relative_residual
from e00047_score import by_series, match_event

SEED: Final[int] = 43
"""Experiment 43's seed: batch `i` is injected with `SEED + i`."""
SETTINGS: Final[dict[str, tuple[DetectorType, int]]] = {
    "D0_8": ("D0", 8),
    "D1_8": ("D1", 8),
    "D0_48": ("D0", 48),
    "D1_48": ("D1", 48),
}
"""The four detector settings: detector and window in half-hours (4 h and 24 h)."""
OLD_THRESHOLDS: Final[dict[str, float]] = {
    "D0_8": 6.0,
    "D1_8": 3.75,
    "D0_48": 4.0,
    "D1_48": 2.5,
}
"""Experiment 43's thresholds, in units of each series' odd-month noise scale."""
TARGET_COUNTS: Final[dict[str, int]] = {"D0_8": 120, "D1_8": 118, "D0_48": 80, "D1_48": 36}
"""H0.5 seed 0's odd-month unexplained-alarm counts at experiment 43's thresholds."""
GRID_STEP: Final[float] = 0.05
COARSE_STEP: Final[float] = 0.25
SCAN_START_ABOVE: Final[float] = 1.5
SCAN_FLOOR: Final[float] = 1.0
EXPLAIN_PAD: Final[int] = 48
"""An alarm within 24 hours of a logged closure on its series is explained by that closure."""
VARIANTS: Final[tuple[tuple[str, bool], ...]] = (("plain", False), ("fixed", True))
CHRISTMAS_NAMES: Final[tuple[str, ...]] = ("Christmas", "Boxing", "New Year")
SAMPLE_NAMES: Final[dict[str, str]] = {
    "D0_8": "alarms_D0_8_6.0",
    "D1_8": "alarms_D1_8_3.75",
    "D0_48": "alarms_D0_48_4.0",
    "D1_48": "alarms_D1_48_2.5",
}
"""Experiment 43's setting names, the keys of its alarm sample and event tables."""

Arrays = dict[int, np.ndarray]


@dataclass(frozen=True)
class ArmPanel:
    """Power, expected power, and residual on the half-hour grid for one arm-seed."""

    name: str
    power: Arrays
    expected: Arrays
    residual: Arrays


def capacities() -> dict[int, float]:
    """Return each series' median effective capacity in MW."""
    table = (
        pl.scan_parquet(FEATURES_PATH)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median())
        .collect()
    )
    return {int(s): float(c) for s, c in table.iter_rows()}


def arm_name(*, arm: str, seed: int) -> str:
    """Return the name of an arm-seed, such as `HOL_s0`."""
    return f"{arm}_s{seed}"


def stored_panel() -> ArmPanel:
    """Return `H0.5` seed 0 as experiment 37 stored it (experiment 17's fit)."""
    panel = load_panel()
    return ArmPanel("H0.5_s0", panel["power"], panel["expected"], panel["residual"])


def load_arm_panel(*, name: str, fits_dir: Path, fallback: bool = False) -> ArmPanel:
    """Return the panel of an arm-seed from its fitted predictions.

    `H0.5_s0` is the stored panel. For the others, the expected power is the prediction times the
    series' capacity, and the observed power is the stored panel's. The prediction must cover every
    stamp the stored panel covers; with `fallback` (the smoke test only), a missing stamp takes the
    stored expected power instead.
    """
    stored = stored_panel()
    if name == "H0.5_s0":
        return stored
    capacity = capacities()
    expected: Arrays = {}
    for series in SERIES_IDS:
        path = fits_dir / name / f"{series}.parquet"
        array = np.full(GRID_LENGTH, np.nan)
        if path.exists():
            frame = pl.read_parquet(path).with_columns(
                stamp=((pl.col("valid_time").dt.epoch("s") - int(GRID_START.timestamp())) // 1800)
            )
            frame = frame.filter(pl.col("stamp").is_between(0, GRID_LENGTH - 1))
            array[frame["stamp"].to_numpy()] = (
                frame["prediction"].to_numpy().astype(np.float64) * capacity[series]
            )
        elif not fallback:
            raise FileNotFoundError(path)
        missing = np.isfinite(stored.expected[series]) & ~np.isfinite(array)
        if missing.any():
            if not fallback:
                raise ValueError(f"{name} series {series}: {missing.sum()} stamps without a fit")
            array[missing] = stored.expected[series][missing]
        array[~np.isfinite(stored.expected[series])] = np.nan
        expected[series] = array
    residual = {s: stored.power[s] - expected[s] for s in SERIES_IDS}
    return ArmPanel(name, stored.power, expected, residual)


def bank_holidays_outside_christmas() -> np.ndarray:
    """Return the bank holiday dates in England other than Christmas, Boxing Day, and New Year."""
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    kept = [d for d, n in calendar.items() if not any(word in n for word in CHRISTMAS_NAMES)]
    return np.array(sorted(kept), dtype="datetime64[D]")


def bank_holiday_distance_all() -> pl.DataFrame:
    """Return `date` and `days_from_holiday` (positive after the holiday) for every calendar date.

    Every bank holiday in England counts, Christmas Day, Boxing Day, and New Year's Day included.
    """
    dates = np.array(bank_holiday_dates_all(), dtype="datetime64[D]")
    days = pl.date_range(
        date(HOLIDAY_YEARS.start, 1, 1), date(HOLIDAY_YEARS.stop - 1, 12, 31), eager=True
    )
    gaps = (days.to_numpy()[:, None] - dates[None, :]).astype(int)
    nearest = gaps[np.arange(days.len()), np.abs(gaps).argmin(axis=1)]
    return pl.DataFrame({"date": days, "days_from_holiday": nearest})


def stamp_dates() -> np.ndarray:
    """Return the local (Europe/London) calendar date of every grid stamp, as `datetime64[D]`."""
    index = pl.datetime_range(
        GRID_START,
        GRID_START + timedelta(minutes=30 * (GRID_LENGTH - 1)),
        "30m",
        eager=True,
        time_zone="UTC",
    )
    return index.dt.convert_time_zone("Europe/London").dt.date().to_numpy()


def holiday_distance_by_stamp() -> np.ndarray:
    """Return, for every grid stamp, the signed days from its local date to the nearest holiday.

    Positive after the holiday. Only bank holidays outside Christmas and New Year count.
    """
    dates = stamp_dates().astype("datetime64[D]")
    holiday_dates = bank_holidays_outside_christmas()
    unique = np.unique(dates)
    gaps = (unique[:, None] - holiday_dates[None, :]).astype(int)
    nearest = gaps[np.arange(len(unique)), np.abs(gaps).argmin(axis=1)]
    return nearest[np.searchsorted(unique, dates)]


@dataclass
class ArmInputs:
    """Detector inputs of one arm-seed's uninjected residual, for both windows."""

    inputs: dict[int, DetectorInputs]

    @staticmethod
    def of(*, residual: Arrays) -> ArmInputs:
        """Compute level shifts and odd-month noise scales for the 4-hour and 24-hour windows."""
        return ArmInputs({w: prepare(residuals=residual, window=w) for w in (8, 48)})

    def scales(self) -> dict[int, dict[int, float]]:
        """The noise scale per window and series."""
        return {w: i.scale for w, i in self.inputs.items()}


def alarms_at(*, arm_inputs: ArmInputs, setting: str, threshold: float) -> pl.DataFrame:
    """Return the alarms of one setting at one threshold."""
    detector, window = SETTINGS[setting]
    return alarms(inputs=arm_inputs.inputs[window], detector=detector, threshold=threshold)


def explained_flags(*, table: pl.DataFrame, logged: dict[int, np.ndarray]) -> np.ndarray:
    """Whether each alarm lies within 24 hours of a logged closure on its series."""
    return np.array([bool(logged[s][t]) for s, t in table.select("series", "stamp").iter_rows()])


def odd_unexplained(*, table: pl.DataFrame, logged: dict[int, np.ndarray]) -> int:
    """Count the alarms in odd months that no logged closure explains."""
    if table.is_empty():
        return 0
    odd = np.isin(stamp_months()[table["stamp"].to_numpy()], list(TUNING_MONTHS))
    return int((odd & ~explained_flags(table=table, logged=logged)).sum())


def tune_setting(
    *, arm_inputs: ArmInputs, setting: str, target: int, logged: dict[int, np.ndarray]
) -> tuple[float, list[tuple[float, int]]]:
    """Return the lowest threshold on a 0.05 grid with at most `target` alarms.

    The count is of unexplained alarms in odd months.

    The scan moves down from experiment 43's threshold plus 1.5 in steps of 0.25 to the first count
    above `target`, then up in steps of 0.05 from there. The second return value holds every
    (threshold, count) pair evaluated, for the operating-point figure.
    """
    curve: dict[float, int] = {}

    def count(threshold: float) -> int:
        key = round(threshold, 2)
        if key not in curve:
            curve[key] = odd_unexplained(
                table=alarms_at(arm_inputs=arm_inputs, setting=setting, threshold=key),
                logged=logged,
            )
        return curve[key]

    threshold = round(OLD_THRESHOLDS[setting] + SCAN_START_ABOVE, 2)
    while threshold > SCAN_FLOOR and count(threshold) <= target:
        threshold = round(threshold - COARSE_STEP, 2)
    threshold = max(threshold, SCAN_FLOOR)
    while count(threshold) > target:
        threshold = round(threshold + GRID_STEP, 2)
    # now `threshold` satisfies the target; step down to the lowest value of the contiguous run
    while threshold - GRID_STEP >= SCAN_FLOOR and count(threshold - GRID_STEP) <= target:
        threshold = round(threshold - GRID_STEP, 2)
    return threshold, sorted(curve.items())


def detect_all(
    *,
    residual: Arrays,
    expected: Arrays,
    scales: dict[int, dict[int, float]],
    thresholds: dict[str, float],
) -> dict[tuple[str, str], tuple[pl.DataFrame, list]]:
    """Alarms and intervals for each setting and interval variant, at the given thresholds."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = {w: prepare(residuals=residual, window=w, scale=scales.get(w)) for w in (8, 48)}
    out: dict[tuple[str, str], tuple[pl.DataFrame, list]] = {}
    for setting, (detector, window) in SETTINGS.items():
        table = alarms(inputs=inputs[window], detector=detector, threshold=thresholds[setting])
        for variant, fixed in VARIANTS:
            intervals = build_intervals(
                alarm_table=table,
                relative=relative,
                window=window,
                detector=detector,
                fixed=fixed,
            )
            out[(setting, variant)] = (table, intervals)
    return out


def drawn_events(*, power: Arrays, expected_path: Path | None = None) -> list[Event]:
    """Redraw experiment 43's 600 events and check they equal its stored table."""
    quiet = near_log_masks(pad_stamps=QUIET_STAMPS)
    for name in OPERATING_POINTS:
        table = pl.read_parquet(EXPERIMENT_43_DIR / f"baseline_alarms_{name}.parquet")
        for series, stamp in table.select("series", "stamp").iter_rows():
            quiet[series][max(stamp - 2 * 48, 0) : stamp + 2 * 48] = True
    events = draw_events(quiet=quiet, power=power, seed=SEED)
    stored = pl.read_parquet(expected_path or EXPERIMENT_43_DIR / "events.parquet")
    fresh = events_frame(events=events)
    if not fresh.equals(stored):
        raise ValueError("the redrawn events differ from experiment 43's events.parquet")
    return events


_STATE: dict = {}


def init_worker(state: dict) -> None:
    """Pool initializer: share the panel and thresholds with forked workers."""
    _STATE.update(state)


def score_batch(args: tuple[int, list[Event]]) -> list[dict]:
    """Inject one batch with the fast residual, detect, and return one row per event and setting."""
    index, batch = args
    panel: ArmPanel = _STATE["panel"]
    injection = inject_power(power=panel.power, events=batch, seed=SEED + index)
    residual = injection.fast_residual(panel.residual)
    results = detect_all(
        residual=residual,
        expected=panel.expected,
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
    )
    return rows_for_events(batch=batch, results=results)


def rows_for_events(
    *, batch: list[Event], results: dict[tuple[str, str], tuple[pl.DataFrame, list]]
) -> list[dict]:
    """Match every event of a batch against each setting's and variant's intervals."""
    rows: list[dict] = []
    for (setting, variant), (_, intervals) in results.items():
        grouped = by_series(intervals=intervals)
        for event in batch:
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=grouped
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "setting": setting,
                    "variant": variant,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                    "known_delay": match.known_delay,
                }
            )
    return rows


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
    return pl.DataFrame(rows)


def score_real(
    *, results: dict[tuple[str, str], tuple[pl.DataFrame, list]], real: pl.DataFrame
) -> pl.DataFrame:
    """Match the labelled real transfers against each setting's intervals on the real residual."""
    rows = []
    for (setting, variant), (_, intervals) in results.items():
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
                    "variant": variant,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                }
            )
    return pl.DataFrame(rows)


def alarm_rows(
    *, results: dict[tuple[str, str], tuple[pl.DataFrame, list]], logged: dict[int, np.ndarray]
) -> pl.DataFrame:
    """One row per real alarm (plain variant), flagged for logged explanation, parity, holidays."""
    months = stamp_months()
    distance = holiday_distance_by_stamp()
    pieces = []
    for (setting, variant), (table, _) in results.items():
        if variant != "plain" or table.is_empty():
            continue
        stamps = table["stamp"].to_numpy()
        pieces.append(
            table.select("series", "stamp", "step_mw", "z").with_columns(
                setting=pl.lit(setting),
                logged_matched=pl.Series(explained_flags(table=table, logged=logged)),
                month=pl.Series(months[stamps]),
                holiday_distance=pl.Series(distance[stamps]),
            )
        )
    return pl.concat(pieces)
