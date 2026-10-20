"""Constants and helpers shared by the scripts of experiment 116 (differentiable physics, NRA).

The experiment fits the convex model of `studies.dp_gam` per substation at three cuts (the tuning
cut 2023-07-01, the replicate cut 2024-07-01, and the scored cut 2025-07-01), fits flows on the
interconnection edges between Primary substations to its residuals (`studies.dp_edge_flow`), and
scores injected transfers. The private interconnection file is read at run time: no edge list is
written into this repository, except the two edges the gate and C3 cover.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import GamSpec
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 116
SLUG: Final[str] = "differentiable-physics-nra-edge-flows-between-primaries"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (fits, tuning, reports); private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""
PART_4_TUNED: Final[Path] = CACHE_DIR.parent / "work" / "tuned.json"
HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
"""Private topology files; never commit anything read from here."""
SERIES_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""MW Primaries that carry the verdicts; D9 is reported beside them."""
GATE_EDGES: Final[tuple[tuple[str, str], ...]] = (("D5", "D16"), ("D10", "D11"))
"""The two MW-MW edges with data in the tuning and replicate years: the gate and C3 cover them."""
GATE_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D16")
GENERATOR_SERIES: Final[tuple[str, ...]] = ("D15", "D18")
"""The two series labelled G-c on the page; every edge that touches them is left out of the
decision graph. They are never named beside G-c in anything published."""
PLACEBO_PAIRS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "tuning": (("D5", "D10"), ("D16", "D14")),
    "replicate": (("D5", "D11"), ("D16", "D10")),
}
"""Pairs with no interconnection: fitted in separate runs, never part of the decision graph."""

TUNING_CUT: Final[datetime] = datetime(2023, 7, 1, tzinfo=UTC)
REPLICATE_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SCORED_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
CUTS: Final[dict[str, datetime]] = {
    "tuning": TUNING_CUT,
    "replicate": REPLICATE_CUT,
    "scored": SCORED_CUT,
}
SEASON_YEARS: Final[range] = range(2019, 2026)
WINDOW_MONTHS: Final[int] = 24
"""GAM-K trains on the 24 months before its cut."""
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty."""
HALF_HOURS_PER_YEAR: Final[int] = 17_520
MIN_TRAIN_MONTHS: Final[float] = 6.0
"""A site with fewer months of usable rows before a cut is left out of that cut."""

D10_JUMP_WINDOWS: Final[tuple[tuple[datetime, datetime], ...]] = (
    (datetime(2024, 11, 8, tzinfo=UTC), datetime(2024, 12, 2, tzinfo=UTC)),
    (datetime(2025, 11, 17, tzinfo=UTC), datetime(2025, 12, 15, tzinfo=UTC)),
)
"""D10's two unlabelled jumps, 2024-11-08 to 2024-12-01 and 2025-11-17 to 2025-12-14, inclusive."""


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def to_64(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`."""
    return np.datetime64(moment.replace(tzinfo=None), "us")


def year_end(*, cut: datetime) -> datetime:
    """Return the end of the scored year that starts at `cut`, never past the held-out start."""
    return min(cut.replace(year=cut.year + 1), HOLDOUT_START)


def cut_label(*, cut: datetime) -> str:
    """Return a cut's date as `YYYY-MM-DD`."""
    return cut.date().isoformat()


def year_slice(*, cache: FeatureCache, cut: datetime) -> slice:
    """Return the grid rows of the scored year that starts at `cut`: ends after the cut."""
    after = np.searchsorted(cache.end_times, to_64(moment=cut), side="right")
    through = np.searchsorted(
        cache.end_times,
        to_64(moment=min(year_end(cut=cut), HOLDOUT_START - timedelta(minutes=30))),
        side="right",
    )
    return slice(int(after), int(through))


def window_slice(*, cache: FeatureCache, cut: datetime) -> slice:
    """Return the grid rows from the start of the 24-month training window to the year's end."""
    start = cut.replace(year=cut.year - WINDOW_MONTHS // 12)
    first = np.searchsorted(cache.end_times, to_64(moment=start), side="right")
    return slice(int(first), year_slice(cache=cache, cut=cut).stop)


def gam_spec(*, units: str) -> GamSpec:
    """Return GAM-K's penalties: no drift, the tuned ridge and step penalty, pinned capacity.

    An MVA substation trains on dark rows only, so its capacity is fitted and has no signal.
    """
    import json

    settings = json.loads(PART_4_TUNED.read_text())
    return GamSpec(
        drift="none",
        ridge=RIDGE,
        drift_step=settings["drift_step"],
        step_penalty=settings["step_penalty"],
        capacity="fitted" if units == "MVA" else "pinned",
        final_knot="free",
    )


def private_edges() -> tuple[tuple[str, str], ...]:
    """Read the private interconnection file and return its distinct pairs as D-label pairs.

    The pair of any series that the series table does not hold is dropped.

    Returns:
        Sorted pairs, the smaller label first.
    """
    series = pl.read_parquet(SERIES_PATH).select("time_series_id", "site")
    monitors = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        "TimeSeriesID", "SubstationNumber"
    )
    label = {
        row[0]: row[1]
        for row in series.join(monitors, left_on="time_series_id", right_on="TimeSeriesID")
        .select("SubstationNumber", "site")
        .rows()
    }
    pairs = pl.read_csv(HIERARCHY_DIR / "Primary Substation Interconnections.csv").rows()
    return tuple(
        sorted(
            {
                tuple(sorted((label[a], label[b])))
                for a, b in pairs
                if a in label and b in label and label[a] != label[b]
            }
        )
    )
