"""Stage 1: tune k on the real stream, then run the pair alarm on every injected world.

```bash
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python e00162_s_detect_run.py tune|worlds|real
```
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00162_s_alarm as alarm
import numpy as np
import polars as pl
from e00162_h_shaped import inject_arm
from e00162_s_d_common import GRID_LENGTH, GRID_START, WORK_DIR, near_log_masks
from e00162_s_d_detect import quiet_masks, seeded_batches
from e00162_s_d_detectors import DetectorInputs
from e00162_s_d_inject import Event, draw_events
from e00162_s_d_variants import load_both, scales_for, tracked_relative, variant_inputs

N_WORKERS: Final[int] = 4
SEED_FRESH_BASE: Final[int] = 8200
FRESH_BASE_EVENTS: Final[int] = 1160
FRESH_BASE_FIRST_ID: Final[int] = 40_000
ARM: Final[str] = os.environ["E162_ARM"]
"""The injection arm: SHARE, CONST, DEMAND, SOLAR, or MIX."""
WORLDS_141: Final[Path] = WORK_DIR / "worlds" / ARM
OUT: Final[Path] = WORK_DIR / "detect" / ARM
SITE_SERIES: Final[tuple[int, ...]] = (17, 10, 8, 16, 18, 7)
FIRST_STAMP: Final[int] = int(
    (np.datetime64("2024-07-01") - np.datetime64(GRID_START.replace(tzinfo=None)))
    .astype("timedelta64[m]")
    .astype(int)
    // 30
)
_STATE: dict = {}


def state() -> dict:
    """Panels and the odd-month noise unit of the pair statistic at every window."""
    if not _STATE:
        panel, offset_panel = load_both()
        kwargs = {"residual": panel["residual"], "offset": offset_panel["residual"]}
        _STATE.update(
            panel=panel,
            offset_panel=offset_panel,
            scales={w: scales_for(variant="X", window=w, **kwargs) for w in (*alarm.WINDOWS, 48)},
        )
    return _STATE


def inputs_for(*, residual: dict, offset: dict, window: int) -> DetectorInputs:
    """The step statistic of every series at one window."""
    return variant_inputs(
        variant="X",
        residual=residual,
        offset=offset,
        scale=state()["scales"][window],
        window=window,
    )


def world_events() -> list[tuple[int, list[Event]]]:
    """The batches of experiment 141's worlds, redrawn with their receivers."""
    panel = state()["panel"]
    base = draw_events(
        quiet=quiet_masks(),
        power=panel["power"],
        seed=SEED_FRESH_BASE,
        headline_only=FRESH_BASE_EVENTS,
        first_id=FRESH_BASE_FIRST_ID,
    )
    sites = set(SITE_SERIES)
    keep = {int(p.stem.split("_")[1]) for p in WORLDS_141.glob("world_*.parquet")}
    groups = seeded_batches(events=base, seed=SEED_FRESH_BASE + 1000)
    out = [(s, b) for s, b in groups if s in keep]
    assert all(any(e.onset >= FIRST_STAMP and sites & set(e.series) for e in b) for _, b in out), (
        "a stored world has no scored event"
    )
    return out


def tune() -> None:
    """Real alarms in odd months that no logged closure explains, for every window and k."""
    panel, offset_panel = state()["panel"], state()["offset_panel"]
    logged = near_log_masks(pad_stamps=alarm.EXPLAIN_PAD)
    rows = []
    for window in alarm.WINDOWS:
        inputs = inputs_for(
            residual=panel["residual"], offset=offset_panel["residual"], window=window
        )
        for k in alarm.K_GRID:
            table = alarm.pair_alarms(inputs=inputs, k=k)
            rows.append(
                {
                    "window": window,
                    "k": k,
                    "alarms": table.height,
                    "unexplained_odd": alarm.unexplained_odd(table=table, logged=logged),
                }
            )
            print(rows[-1], flush=True)
    frame = pl.DataFrame(rows)
    chosen = {}
    for window in alarm.WINDOWS:
        ok = frame.filter(
            (pl.col("window") == window) & (pl.col("unexplained_odd") <= alarm.FALSE_ALARM_TARGET)
        )
        chosen[str(window)] = float(ok["k"].min()) if ok.height else None  # ty: ignore[invalid-argument-type]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT / "tune_grid.parquet")
    (OUT / "chosen_k.json").write_text(json.dumps(chosen))
    print("matched k by window:", chosen)


def world_job(args: tuple[int, list[Event]]) -> str:
    """Inject one batch and write the pair-alarm intervals and event scores at every window."""
    seed, batch = args
    path = OUT / f"world_{seed}_events.parquet"
    if path.exists():
        return str(path)
    s = state()
    chosen = json.loads((OUT / "chosen_k.json").read_text())
    injection, _ = inject_arm(power=s["panel"]["power"], events=batch, seed=seed, arm=ARM)  # ty: ignore[invalid-argument-type]
    residual = injection.fast_residual(s["panel"]["residual"])
    offset = injection.fast_residual(s["offset_panel"]["residual"])
    relative = tracked_relative(
        variant="X",
        panel=s["panel"],
        offset_panel=s["offset_panel"],
        residual=residual,
        offset=offset,
    )
    fast24 = inputs_for(residual=residual, offset=offset, window=48)
    moved = alarm.moved_mw_of(events=batch, delta=injection.delta)
    ronc = alarm.with_ronc_onset_known(
        intervals=pl.read_parquet(WORLDS_141 / f"world_{seed}.parquet").filter(
            pl.col("setting") == "R0_M48"
        )
    )
    scored = [alarm.score_events(events=batch, moved_mw=moved, intervals=ronc, family="RONC")]
    for window in alarm.WINDOWS:
        k = chosen[str(window)]
        if k is None:
            continue
        table = alarm.pair_alarms(
            inputs=inputs_for(residual=residual, offset=offset, window=window), k=k
        )
        pair = alarm.intervals_of(table=table, relative=relative, fast24=fast24, world=seed)
        union = alarm.union_intervals(ronc=ronc, pair=pair)
        pair.write_parquet(OUT / f"world_{seed}_W{window}_pair.parquet")
        union.write_parquet(OUT / f"world_{seed}_W{window}_union.parquet")
        scored.append(
            alarm.score_events(events=batch, moved_mw=moved, intervals=pair, family=f"NEW{window}")
        )
        scored.append(
            alarm.score_events(
                events=batch, moved_mw=moved, intervals=union, family=f"UNION{window}"
            )
        )
    pl.concat(scored).with_columns(world=pl.lit(seed)).write_parquet(path)
    return str(path)


def worlds() -> None:
    """Run every world in a pool of workers."""
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = world_events()
    print(f"{len(jobs)} worlds", flush=True)
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        done = list(pool.map(world_job, jobs))
    print(f"{len(done)} worlds written")


def real() -> None:
    """The pair alarm and its intervals on the real stream, at every window's matched k."""
    s = state()
    chosen = json.loads((OUT / "chosen_k.json").read_text())
    panel, offset_panel = s["panel"], s["offset_panel"]
    relative = tracked_relative(
        variant="X",
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
    )
    fast24 = inputs_for(residual=panel["residual"], offset=offset_panel["residual"], window=48)
    for window in alarm.WINDOWS:
        k = chosen[str(window)]
        if k is None:
            continue
        inputs = inputs_for(
            residual=panel["residual"], offset=offset_panel["residual"], window=window
        )
        table = alarm.pair_alarms(inputs=inputs, k=k)
        table.write_parquet(OUT / f"real_W{window}_alarms.parquet")
        alarm.intervals_of(table=table, relative=relative, fast24=fast24, world=0).write_parquet(
            OUT / f"real_W{window}_pair.parquet"
        )
        print(window, k, table.height, flush=True)


def main() -> None:
    """Dispatch a stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tune", "worlds", "real"])
    args = parser.parse_args()
    {"tune": tune, "worlds": worlds, "real": real}[args.stage]()
    assert GRID_LENGTH > 0


if __name__ == "__main__":
    main()
