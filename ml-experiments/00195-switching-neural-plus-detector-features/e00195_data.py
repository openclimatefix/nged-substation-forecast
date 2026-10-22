# ruff: noqa: E501
"""Candidate tables for the real record and for injected worlds, with labels and the 17 block days.

A world is one draw of up to 200 injected transfers (experiment 189's injector, fitted without a fold's
Primary pairs) added to the residual of both convex models. Every candidate gets a label: the id of the
injected event whose onset it sits within 2 hours of, 0 for a clean negative, and -1 when it must not
be used for training (inside an injected span, explained by a logged or labelled real event, inside a
D8 drop, or on one of the 17 block days).

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00195-*/e00195_data.py FOLD
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00195_common import (
    BG2_PREDICTIONS,
    GRID_LENGTH,
    GRID_START,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    WORK_DIR,
    explained_masks,
    load_panel,
    noise_units,
    pair_folds,
)
from e00195_injector import fit_injector
from e00195_ronc import (
    Scales,
    candidates,
    load_offset_panel,
    make_scales,
    make_stats,
)
from e00195_worlds import Calendar, disjoint_world, inject

EVEN_MONTHS: Final[frozenset[int]] = frozenset(set(range(1, 13)) - TUNING_MONTHS)
N_EVENTS: Final[int] = 200
WORLD_DIR: Final = WORK_DIR / "worlds"
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

    def world(self, *, fit, seed: int, allow_months: frozenset[int] | None):  # noqa: ANN001, ANN201
        """Draw events, inject them into both residuals, and return stats, candidates, events."""
        rng = np.random.default_rng(seed)
        events = disjoint_world(
            fit=fit,
            n_events=N_EVENTS,
            residual=self.panel["residual"],
            power=self.panel["power"],
            sigma=self.sigma,
            calendar=self.calendar,
            rng=rng,
            allow_months=allow_months,
        )
        residual = inject(residual=self.panel["residual"], events=events)
        offset = inject(residual=self.offset_panel["residual"], events=events)
        stats = make_stats(
            residual=residual,
            offset=offset,
            scales=self.scales,
            panel=self.panel,
            offset_panel=self.offset_panel,
        )
        return stats, candidates(stats=stats, explained=self.explained), events


def label(*, table: pl.DataFrame, events: list, base: Base) -> pl.DataFrame:
    """Add `event_id` (index of the injected event this candidate marks, else -1) and `label`."""
    n = table.height
    event_id = np.full(n, -1, dtype=int)
    in_span = np.zeros(n, dtype=bool)
    series, stamp = table["series"].to_numpy(), table["stamp"].to_numpy()
    for k, placed in enumerate(events):
        on_series = np.isin(series, placed.series)
        near = on_series & (np.abs(stamp - placed.onset) <= 4)
        event_id[near & (event_id < 0)] = k
        in_span |= on_series & (stamp >= placed.onset - 4) & (stamp <= placed.end + SPAN_PAD)
    blocked = np.array(
        [
            bool(base.blocks.get(s, np.zeros(GRID_LENGTH, bool))[t])
            for s, t in zip(series, stamp, strict=True)
        ]
    )
    unusable = table["explained"].to_numpy() | table["d8_drop"].to_numpy() | blocked
    lab = np.where(event_id >= 0, 1, np.where(in_span | unusable, -1, 0))
    return table.with_columns(
        event_id=pl.Series(event_id), label=pl.Series(lab), blocked=pl.Series(blocked)
    )


def events_frame(*, events: list) -> pl.DataFrame:
    """One row per injected event: series, timing, depth, shape, and how many receivers are metered."""
    return pl.DataFrame(
        [
            {
                "event_id": k,
                "donor": p.event.donor,
                "series": list(p.series),
                "onset": p.onset,
                "end": p.end,
                "hours": (p.end - p.onset) / 2,
                "depth_z": p.event.depth_z,
                "shape": p.event.shape,
                "n_receivers": len(p.event.receivers),
            }
            for k, p in enumerate(events)
        ]
    )


_BASE: dict = {}


def _build(job: tuple[int, str, int, int]) -> str:
    """Build and save one world; returns its stem."""
    fold, kind, index, seed = job
    if "base" not in _BASE:
        _BASE["base"] = Base()
        _BASE["folds"] = pair_folds()
    base: Base = _BASE["base"]
    held = frozenset(p for p, f in _BASE["folds"].items() if f == fold)
    fit = fit_injector(held_out_pairs=held, sigma=base.sigma)
    held_tuples = tuple(tuple(int(s) for s in p.split(",")) for p in held)
    use = replace(fit, pairs=held_tuples) if KINDS[kind][1] == "held" else fit
    months = frozenset(range(1, 13)) - EVEN_MONTHS if kind.startswith("eval") else EVEN_MONTHS
    stats, cands, events = base.world(fit=use, seed=seed, allow_months=frozenset(months))
    del stats
    table = label(table=cands.table, events=events, base=base)
    stem = f"f{fold}_{kind}_{index}"
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(WORLD_DIR / f"{stem}.parquet")
    events_frame(events=events).write_parquet(WORLD_DIR / f"{stem}_events.parquet")
    np.save(WORLD_DIR / f"{stem}_windows.npy", cands.windows.astype(np.float16))
    return stem


def main(*, folds: list[int], workers: int = 4) -> None:
    """Build every world of the given folds, `workers` at a time."""
    jobs = [
        (fold, kind, i, 195_000 + 1000 * fold + 100 * list(KINDS).index(kind) + i)
        for fold in folds
        for kind, (count, _) in KINDS.items()
        for i in range(count)
        if not (WORLD_DIR / f"f{fold}_{kind}_{i}.parquet").exists()
    ]
    with ProcessPoolExecutor(workers) as pool:
        for stem in pool.map(_build, jobs):
            print("built", stem, flush=True)


if __name__ == "__main__":
    main(folds=[int(a) for a in sys.argv[1:]] or [0, 1, 2])
