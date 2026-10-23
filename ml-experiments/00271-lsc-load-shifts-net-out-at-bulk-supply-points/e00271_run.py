r"""Score the real record and every injected world, one parquet file per world.

Run (3 workers, thread caps 1, memory capped):

```bash
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \\
    uv run --no-sync python e00271_run.py 40 3
```

The first argument is the number of worlds per kind and placement, the second the workers. A
world whose files exist is skipped, so a crashed run resumes. Each file is written atomically.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Final

import e00271_inputs as inputs
import e00271_score as sc
import e00271_worlds as worlds
import polars as pl

WORLD_DIR: Final[Path] = inputs.RESULTS_DIR / "worlds"
_CTX: sc.Context | None = None


def _init() -> None:
    global _CTX  # noqa: PLW0603
    _CTX = sc.build_context()


def _write(*, frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)


def run_world(job: tuple[str, str, int]) -> str:
    """Draw, push up, and score one world."""
    kind, placement, index = job
    stem = f"{kind}_{placement}_{index:03d}"
    if (WORLD_DIR / f"{stem}_quality.parquet").exists():
        return stem
    ctx = _CTX
    if ctx is None:
        msg = "worker not initialised"
        raise RuntimeError(msg)
    shifts = worlds.draw_world(
        kind=kind, placement=placement, index=index, event_days=ctx.event_days
    )
    forecast, observed = worlds.node_deltas(shifts=shifts)
    cost, quality = sc.score_world(ctx=ctx, shifts=shifts, forecast=forecast, observed=observed)
    bsp = worlds.bsp_of_series()
    table = pl.DataFrame(
        {
            "donor": [s.donor for s in shifts],
            "receiver": [s.receiver for s in shifts],
            "onset": [s.onset for s in shifts],
            "end": [s.end for s in shifts],
            "moved_mw": [s.moved_mw for s in shifts],
            "detected": [s.detected for s in shifts],
            "crosses": [bsp[s.donor] != bsp[s.receiver] for s in shifts],
        }
    )
    _write(frame=table, path=WORLD_DIR / f"{stem}_shifts.parquet")
    _write(frame=cost, path=WORLD_DIR / f"{stem}_cost.parquet")
    _write(frame=quality, path=WORLD_DIR / f"{stem}_quality.parquet")
    return stem


def main(n_worlds: int, workers: int) -> None:
    """Score the real record, then every world."""
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _init()
    ctx = _CTX
    if ctx is not None and not (WORLD_DIR / "real_quality.parquet").exists():
        forecast, observed = worlds.node_deltas(shifts=[])
        cost, quality = sc.score_world(ctx=ctx, shifts=[], forecast=forecast, observed=observed)
        _write(frame=cost, path=WORLD_DIR / "real_cost.parquet")
        _write(frame=quality, path=WORLD_DIR / "real_quality.parquet")
    jobs = [
        (kind, placement, i)
        for kind in worlds.KINDS
        for placement in worlds.PLACEMENTS
        for i in range(n_worlds)
    ]
    with mp.get_context("spawn").Pool(workers, initializer=_init, maxtasksperchild=20) as pool:
        for k, stem in enumerate(pool.imap_unordered(run_world, jobs)):
            print(f"{k + 1}/{len(jobs)} {stem} {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
