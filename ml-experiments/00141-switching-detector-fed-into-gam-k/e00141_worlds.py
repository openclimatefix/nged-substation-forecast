"""Injected worlds: transfers of known size and timing, and the detector's causal alarms on them.

Reuses experiment 147's headline-band draw (the same events and seeds), injects each batch of
non-overlapping events with `inject_power`, runs RO-NC arm B on the fast-path residual, and stores
two interval tables per world: `R2_M48` (the causal live end rule with merging, the deployable
signal) and `R0_M48` (the hindsight end, NOT DEPLOYABLE). The moved megawatts added to each of the
six scored sites are stored beside them.

An interval's onset is known to a forecast only from `onset + ONSET_LAG` (the statistic needs a
full 24-hour window after the step, and a peak test needs 24 hours more), and its end only from
`known_from`.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00141_common import GRID_LENGTH, GRID_START, WORK_DIR
from e00141_detect import seeded_batches
from e00141_inject import inject_power
from e00141_run import (
    _STATE,
    SEED_FRESH_BASE,
    build_detectors,
    draw_fresh,
    make_state,
    run_pool,
)

SITE_SERIES: Final[dict[str, int]] = {"D5": 17, "D10": 10, "D11": 8, "D14": 16, "D16": 18, "D9": 7}
"""The six scored sites and their detector series ids."""
SETTINGS: Final[tuple[str, ...]] = ("R2_M48", "R0_M48")
ONSET_LAG: Final[int] = 96
"""Half-hours between a step's onset and the stamp from which a forecast may know the alarm."""
FIRST_STAMP: Final[int] = int(
    (datetime(2024, 7, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800
)
"""Events starting before the tuning period (2024-07) are not scored."""
OUT: Final = WORK_DIR / "worlds"


def relevant(batch: list) -> bool:
    """True if any event in the batch touches a scored site inside the tuning and scored periods."""
    sites = set(SITE_SERIES.values())
    return any(e.onset >= FIRST_STAMP and sites & set(e.series) for e in batch)


def world_job(args: tuple[int, list]) -> str:
    """Inject one batch, detect, and write the world's intervals and deltas."""
    seed, batch = args
    panel = _STATE["panel"]
    path = OUT / f"world_{seed}.parquet"
    if path.exists():
        return str(path)
    injection = inject_power(power=panel["power"], events=batch, seed=seed)
    results, _ = build_detectors(
        panel=panel,
        offset_panel=_STATE["offset_panel"],
        residual=injection.fast_residual(panel["residual"]),
        offset=injection.fast_residual(_STATE["offset_panel"]["residual"]),
        scales=_STATE["scales"],
        thresholds=_STATE["thresholds"],
        models=_STATE["models"],
    )
    rows = []
    for setting in SETTINGS:
        table, intervals = results[setting]
        steps = {(r["series"], r["stamp"]): r["step_mw"] for r in table.iter_rows(named=True)}
        rows.extend(
            {
                "world": seed,
                "setting": setting,
                "series": iv.series,
                "role": iv.role,
                "alarm_series": iv.alarm_series,
                "onset": iv.onset,
                "end": iv.end,
                "known_from": iv.known_from,
                "step_mw": steps.get((iv.alarm_series, iv.onset), float("nan")),
            }
            for iv in intervals
        )
    delta = {f"delta_{s}": injection.delta[s] for s in SITE_SERIES.values()}
    events = np.array(
        [(e.event_id, e.donor, e.onset, e.end, e.fraction) for e in batch], dtype=np.float64
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"world_{seed}.npz", events=events, **delta)  # ty: ignore[invalid-argument-type]
    pl.DataFrame(rows, schema_overrides={"step_mw": pl.Float64}).write_parquet(path)
    return str(path)


def main() -> None:
    """Draw the events, pick the relevant batches, and write each world."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    state = make_state()
    base, _ = draw_fresh(panel=state["panel"])
    groups = [g for g in seeded_batches(events=base, seed=SEED_FRESH_BASE + 1000) if relevant(g[1])]
    print(f"{len(groups)} relevant batches of {len(base)} events; grid {GRID_LENGTH}", flush=True)
    if args.limit:
        groups = groups[: args.limit]
    done = run_pool(function=world_job, jobs=groups, state=state)
    print(f"{len(done)} worlds written", flush=True)


if __name__ == "__main__":
    main()
