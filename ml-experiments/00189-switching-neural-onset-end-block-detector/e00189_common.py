"""Paths, the half-hourly grid, the convex-model residual panel, and the real events.

The panel and the event tables are those of experiments 130 and 181; this module copies their
loaders so that this experiment imports only its own folder. Nothing here writes outside the
experiment's private folder.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00189-switching-neural-onset-end-block-detector"
"""Private tables and weights; never committed."""
RUN_130: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual"
GAM_PANEL: Final[Path] = RUN_130 / "gam" / "panel_1.pkl"
"""Experiment 130's frozen convex-model panel (model refitted on the 1st of each month)."""
REAL_RESULTS: Final[Path] = RUN_130 / "B" / "real_results.parquet"
PANEL_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "panel.parquet"
LOGGED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "intervals.parquet"
)
D8_DROPS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00159-switching-d8-multi-day-drops-recurrence-and-report"
    / "drops.parquet"
)
BLOCK_ALARMS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00181-switching-block-detector-and-like-for-like-table"
)
BG2_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = int((GRID_END - GRID_START).total_seconds() // 1800)
DAY: Final[int] = 48
"""Half-hours in a day."""
WINDOW: Final[int] = 7 * DAY
"""Half-hours in the network's input window (7 days)."""
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
D8_SERIES: Final[int] = 26
D9_SERIES: Final[int] = 7
TUNING_MONTHS: Final[frozenset[int]] = frozenset({1, 3, 5, 7, 9, 11})
"""Odd months set the alarm threshold; the network trains on even months only."""
ADJACENCY: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}
N_NEIGHBOURS: Final[int] = 4
LOG_PAD: Final[int] = 48
BLANK_SERIES: Final[int] = 12
"""The generator case with no convex-model fit; blanked, as in experiment 130's arm B."""
MAD_TO_STD: Final[float] = 1.4826


def stamp_index() -> pl.Series:
    """The UTC time of every grid stamp."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )


def stamp_months() -> np.ndarray:
    """The UTC calendar month (1 to 12) of every grid stamp."""
    return stamp_index().dt.month().to_numpy()


def stamp_hour_weekday() -> tuple[np.ndarray, np.ndarray]:
    """Local (Europe/London) hour of day as a float and weekday (0 = Monday) of every stamp."""
    local = stamp_index().dt.convert_time_zone("Europe/London")
    hour = local.dt.hour().to_numpy() + local.dt.minute().to_numpy() / 60.0
    return hour, local.dt.weekday().to_numpy() - 1


def holiday_flag() -> np.ndarray:
    """1.0 on every stamp whose local date is an England bank holiday (Christmas included)."""
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028))
    dates = stamp_index().dt.convert_time_zone("Europe/London").dt.date().to_numpy()
    days = {np.datetime64(d) for d in calendar}
    return np.array([np.datetime64(d) in days for d in dates], dtype=np.float32)


def load_panel() -> dict[str, dict[int, np.ndarray]]:
    """The convex-model panel: `power`, `expected`, `residual` per series (NaN where missing).

    `power` comes from experiment 82's panel, the residual and expected power from the frozen
    convex model of experiment 130; series 12 is blanked.
    """
    frame = pl.read_parquet(PANEL_PATH)
    power: dict[int, np.ndarray] = {}
    for (series,), group in frame.group_by(["series"]):
        array = np.full(GRID_LENGTH, np.nan)
        array[group["stamp"].to_numpy()] = group["obs_mw"].to_numpy()
        power[int(series)] = array
    stored = pickle.loads(GAM_PANEL.read_bytes())["frozen"]
    blank = np.full(GRID_LENGTH, np.nan)
    expected = {s: (blank if s == BLANK_SERIES else a) for s, a in stored["expected"].items()}
    residual = {s: (blank if s == BLANK_SERIES else a) for s, a in stored["residual"].items()}
    return {"power": power, "expected": expected, "residual": residual}


def noise_units(*, residual: dict[int, np.ndarray]) -> dict[int, float]:
    """Each series' noise unit: 1.4826 times the MAD of its residual in even-numbered months."""
    months = stamp_months()
    even = ~np.isin(months, list(TUNING_MONTHS))
    out = {}
    for series, values in residual.items():
        chosen = values[even & np.isfinite(values)]
        out[series] = (
            float(MAD_TO_STD * np.median(np.abs(chosen - np.median(chosen))))
            if chosen.size > 1000
            else float("nan")
        )
    return out


def logged_masks(*, pad: int = LOG_PAD) -> dict[int, np.ndarray]:
    """True within `pad` half-hours of a logged closure that touches the series."""
    logged = pl.read_parquet(LOGGED_PATH)
    out = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for row in logged.iter_rows(named=True):
        lo = max(row["first_stamp"] - pad, 0)
        hi = min(row["last_stamp_exclusive"] + pad, GRID_LENGTH)
        for side in (row["side_a"], row["side_b"]):
            if side is not None and side.isdigit() and int(side) in out:
                out[int(side)][lo:hi] = True
    return out


def real_transfers() -> pl.DataFrame:
    """The 66 labelled real transfers of experiment 41, with a `pair` key and series list.

    Columns: card_id, visibility, onset, end, hours, share, series (tuple), pair (string),
    moved_mw_p50, onset_found_ronc and found_ronc (RO-NC on the convex-model residual).
    """
    table = (
        pl.read_parquet(REAL_RESULTS)
        .filter(pl.col("setting") == "XL")
        .select(
            "card_id",
            "visibility",
            "onset",
            "end",
            "hours",
            "share",
            "series",
            "moved_mw_p50",
            onset_found_ronc="onset_found",
            found_ronc="found",
        )
    )
    return table.with_columns(
        pair=pl.col("series"),
        series=pl.col("series").str.split(",").list.eval(pl.element().cast(pl.Int64)),
    )


def labelled_masks() -> dict[int, np.ndarray]:
    """True inside any of the 66 labelled real transfers, on each series the transfer touches."""
    out = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for row in real_transfers().iter_rows(named=True):
        for s in row["series"]:
            if s in out:
                out[s][max(row["onset"], 0) : min(row["end"] + 1, GRID_LENGTH)] = True
    return out


def explained_masks() -> dict[int, np.ndarray]:
    """True within 24 hours of a logged closure or inside a labelled transfer."""
    logged, labelled = logged_masks(), labelled_masks()
    return {s: logged[s] | labelled[s] for s in SERIES_IDS}


def stamp_time(*, stamp: int) -> datetime:
    """UTC start of the half-hour at a stamp."""
    return GRID_START + timedelta(minutes=30 * stamp)


N_FOLDS: Final[int] = 3


def pair_folds() -> dict[str, int]:
    """Assign each Primary pair of the labelled transfers to one of `N_FOLDS` folds.

    Pairs are placed largest first (counting transfers over 1 day fully and shorter ones at a
    tenth) into the fold with the smallest load, so each fold holds about a third of the 35.
    """
    table = real_transfers()
    weight = (
        table.group_by("pair")
        .agg(load=(pl.col("hours") > 24).sum() + 0.1 * (pl.col("hours") <= 24).sum())
        .sort(["load", "pair"], descending=[True, False])
    )
    totals = [0.0] * N_FOLDS
    out: dict[str, int] = {}
    for pair, load in weight.iter_rows():
        fold = int(np.argmin(totals))
        out[pair] = fold
        totals[fold] += load
    return out
