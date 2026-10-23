# ruff: noqa: E501
"""The real record's clean candidates, the 17 block days, and the location of experiment 195's worlds.

Experiment 195 built the worlds (injected transfers added to the residual of both convex models) and
labelled every candidate. This module only reads them (`WORLD_DIR`) and never builds one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00249_common import (
    BG2_PREDICTIONS,
    E195_DIR,
    GRID_LENGTH,
    GRID_START,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    explained_masks,
    load_panel,
    noise_units,
)
from e00249_ronc import (
    Scales,
    candidates,
    load_offset_panel,
    make_scales,
    make_stats,
)
from e00249_worlds import Calendar

EVEN_MONTHS: Final[frozenset[int]] = frozenset(set(range(1, 13)) - TUNING_MONTHS)
N_EVENTS: Final[int] = 200
WORLD_DIR: Final = E195_DIR / "worlds"
KINDS: Final[dict[str, tuple[int, str]]] = {
    "train": (40, "fit"),
    "valid": (8, "fit"),
    "eval_held": (6, "held"),
    "eval_fitted": (3, "fit"),
}
"""Per kind: number of worlds, and whether events use the fold's fitted pairs or its held-out pairs."""
SPAN_PAD: Final[int] = 96
E174_DAYS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00174-dp-unlabelled-worst-days-weather-shared-or-single-site"
    / "work"
    / "worst_days_split.parquet"
)
E175_SHAPES: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work" / "day_shapes.parquet"
)
E175_GRID_FIRST: Final = datetime(2025, 6, 1, tzinfo=UTC)
BLOCK_TOLERANCE: Final[int] = 6


def block_days() -> pl.DataFrame:
    """The 17 unlabelled worst days that experiment 175 classed as sharp-edged blocks."""
    unlabelled = pl.read_parquet(E174_DAYS).filter(~pl.col("labelled")).select("site", "local_date")
    days = (
        pl.read_parquet(E175_SHAPES)
        .join(unlabelled, on=["site", "local_date"])
        .filter(pl.col("shape") == "a_block")
        .sort("site", "local_date")
    )
    pairs = pl.read_parquet(BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    series = {r["site"]: int(r["time_series_id"]) for r in pairs.iter_rows(named=True)}
    offset = int((E175_GRID_FIRST - GRID_START).total_seconds() // 1800)
    start = (
        pl.col("local_date")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("Europe/London")
        .dt.convert_time_zone("UTC")
    )
    days = days.with_columns(
        series=pl.col("site").replace_strict(series, return_dtype=pl.Int64),
        edge_stamp=pl.col("edge_position") + offset,
        first=((start.dt.epoch("s") - int(GRID_START.timestamp())) // 1800).cast(pl.Int64),
    ).with_columns(last=pl.col("first") + 47)
    if days.height != 17:
        raise ValueError(f"expected 17 block days, found {days.height}")
    return days


def block_day_mask() -> dict[int, np.ndarray]:
    """True on every stamp of the 17 block days, per series (kept out of training and selection)."""
    out: dict[int, np.ndarray] = {}
    for d in block_days().iter_rows(named=True):
        out.setdefault(d["series"], np.zeros(GRID_LENGTH, dtype=bool))[
            d["first"] : d["last"] + 1
        ] = True
    return out


class Base:
    """The clean panels, scales, and lookups shared by every world."""

    def __init__(self) -> None:
        """Load the panels and fix the scales on the clean residual."""
        self.panel = load_panel()
        self.offset_panel = load_offset_panel()
        self.sigma = noise_units(residual=self.panel["residual"])
        self.scales: Scales = make_scales(
            residual=self.panel["residual"], offset=self.offset_panel["residual"], sigma=self.sigma
        )
        self.calendar = Calendar()
        self.explained = explained_masks()
        self.blocks = block_day_mask()

    def clean(self):  # noqa: ANN201
        """Candidates of the real, uninjected record."""
        stats = make_stats(
            residual=self.panel["residual"],
            offset=self.offset_panel["residual"],
            scales=self.scales,
            panel=self.panel,
            offset_panel=self.offset_panel,
        )
        return stats, candidates(stats=stats, explained=self.explained)
