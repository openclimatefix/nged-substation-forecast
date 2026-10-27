"""Step 4 of experiment 00356: default 2's head, and the head retrained every month (arm u3).

Both use experiment 00347's mixture head and its `train_and_predict` (arm y2: the 6 plant summaries,
the lead terms, and the one-hot farm), on the table that `e00356_frame.py` stores.

- `d2_h<seed>.npy` (cases, 11): default 2, trained once on the hindcast (runs and valid times
  before 2025-04-16, early stopping on the later hindcast runs), five seeds, predicting every case.
- `u3_s<seed>_<YYYY-MM>.npy` (Tuesday cases of that run month, 11): a head trained on the finished
  runs on the first day of the month: valid times at least 3.5 hours before the month's start, from
  2024-04, early stopping on the last 2 months of that window. Five seeds.

One process per seed keeps the GPU busy (a single fit uses about a quarter of it):

    PYTHONPATH=<see e00356_common> uv run --no-sync python e00356_heads.py --seed 0 [--u3-only]
"""

import os
import sys
import time
from datetime import UTC, datetime

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00347_s4_data as data347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_run as run347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def save(*, name: str, array: np.ndarray) -> None:
    """Write an array to the parts folder atomically."""
    common.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = common.PARTS_DIR / f"{name}.npy"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def month_start(*, year: int, month: int) -> datetime:
    """Return 00:00 UTC on the first day of the month."""
    return datetime(year, month, 1, tzinfo=UTC)


def add_months(*, year: int, month: int, delta: int) -> tuple[int, int]:
    """Return (year, month) shifted by `delta` months."""
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def load() -> tuple[pl.DataFrame, np.ndarray]:
    """Return the case table (`masked` replaced by the complement of `kept`) and the summaries."""
    frame = pl.read_parquet(common.DATA_DIR / "frame.parquet")
    frame = frame.with_columns(masked=~pl.col("kept"))
    return frame, np.load(common.DATA_DIR / "summary.npy")


def train_d2(*, frame: pl.DataFrame, summary: np.ndarray, seeds: tuple[int, ...]) -> None:
    """Train default 2's head once per seed and predict every case."""
    started = time.time()
    hind = (frame["power_fcst_init_time"] < common.FIRST_NEW_RUN).to_numpy()
    for seed in seeds:
        if (common.PARTS_DIR / f"d2_h{seed}.npy").exists():
            continue
        out = run347.train_and_predict(
            hindcast=frame.filter(pl.Series(hind)),
            summary_h=summary[hind],
            rows=np.ones(int(hind.sum()), dtype=bool),
            cutoff=data347.MAIN_CUTOFF,
            valid_limit=None,
            kind="full",
            farm_columns=True,
            targets={"all": (frame, summary)},
            seed=seed,
        )
        save(name=f"d2_h{seed}", array=out["all"])
        print(f"d2 seed {seed} done {time.time() - started:.0f} s", flush=True)


def train_u3(*, frame: pl.DataFrame, summary: np.ndarray, seeds: tuple[int, ...]) -> None:
    """Retrain the head on the first day of every scored month, per seed."""
    started = time.time()
    months, ym = [], common.FIRST_SCORED_MONTH
    while ym <= common.LAST_SCORED_MONTH:
        months.append(ym)
        ym = add_months(year=ym[0], month=ym[1], delta=1)
    for year, month in months:
        label = f"{year}-{month:02d}"
        start = month_start(year=year, month=month)
        window_end = start - common.DELAY
        stop_start = month_start(
            **dict(
                zip(("year", "month"), add_months(year=year, month=month, delta=-2), strict=True)
            )
        )
        finished = (frame["valid_time"] <= window_end).to_numpy()
        target = ((frame["month"] == label) & frame["tuesday"]).to_numpy()
        latest = frame["valid_time"].filter(pl.Series(finished)).max()
        print(f"{label}: latest valid time used {latest}, month start {start}", flush=True)
        if not isinstance(latest, datetime):
            raise TypeError(latest)
        if latest > window_end:
            msg = f"leak: {label} uses valid time {latest}"
            raise ValueError(msg)
        for seed in seeds:
            name = f"u3_s{seed}_{label}"
            if (common.PARTS_DIR / f"{name}.npy").exists():
                continue
            out = run347.train_and_predict(
                hindcast=frame.filter(pl.Series(finished)),
                summary_h=summary[finished],
                rows=np.ones(int(finished.sum()), dtype=bool),
                cutoff=stop_start,
                valid_limit=window_end,
                kind="full",
                farm_columns=True,
                targets={"month": (frame.filter(pl.Series(target)), summary[target])},
                seed=seed,
            )
            save(name=name, array=out["month"])
            print(f"{name} done {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Train default 2's head and the monthly heads."""
    frame, summary = load()
    seeds = (int(sys.argv[sys.argv.index("--seed") + 1]),) if "--seed" in sys.argv else common.SEEDS
    if "--u3-only" not in sys.argv:
        train_d2(frame=frame, summary=summary, seeds=seeds)
    if "--d2-only" not in sys.argv:
        train_u3(frame=frame, summary=summary, seeds=seeds)
    print("all heads done", flush=True)


if __name__ == "__main__":
    main()
