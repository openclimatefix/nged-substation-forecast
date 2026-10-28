"""Step 1 of experiment 00357: the monthly retrained head (arm u3) predicting every kept daily run.

Same heads as experiment 00356's `e00356_heads.py` (trained on the first day of each month on the
finished runs from 2024-04, valid times at least 3.5 hours before the month start, early stopping on
the last 2 months), but each head also predicts every kept case from its own month to the end, so
that one file per head serves both the head's own month (arm u3) and the stale-head placebo.
`u3_s<seed>_<YYYY-MM>.npy` has the cases `case_index()` returns for months from `<YYYY-MM>` on.

    PYTHONPATH=$PYTHONPATH_FOLDERS uv run --no-sync python e00357_heads.py --seed 0
"""

import os
import sys
import time
from datetime import datetime

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00347_s4_run as run347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_common as c356  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_heads as h356  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00357_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def scored_months() -> list[str]:
    """Return every scored month, `YYYY-MM`."""
    out, ym = [], c356.FIRST_SCORED_MONTH
    while ym <= c356.LAST_SCORED_MONTH:
        out.append(f"{ym[0]}-{ym[1]:02d}")
        ym = h356.add_months(year=ym[0], month=ym[1], delta=1)
    return out


def case_mask(*, frame: pl.DataFrame, first_month: str) -> np.ndarray:
    """Return the kept cases whose run month is `first_month` or later."""
    return (frame["kept"] & (frame["month"] >= first_month)).to_numpy()


def save(*, name: str, array: np.ndarray) -> None:
    """Write an array to the parts folder atomically."""
    common.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = common.PARTS_DIR / f"{name}.npy"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def train(*, seed: int) -> None:
    """Retrain the head on the first day of every scored month and predict the later cases."""
    started = time.time()
    frame, summary = h356.load()
    for label in scored_months():
        year, month = int(label[:4]), int(label[5:7])
        start = h356.month_start(year=year, month=month)
        window_end = start - c356.DELAY
        stop_start = h356.month_start(
            **dict(
                zip(
                    ("year", "month"),
                    h356.add_months(year=year, month=month, delta=-2),
                    strict=True,
                )
            )
        )
        finished = (frame["valid_time"] <= window_end).to_numpy()
        latest = frame["valid_time"].filter(pl.Series(finished)).max()
        if not isinstance(latest, datetime) or latest > window_end:
            msg = f"leak: {label} uses valid time {latest}"
            raise ValueError(msg)
        name = f"u3_s{seed}_{label}"
        if (common.PARTS_DIR / f"{name}.npy").exists():
            continue
        target = case_mask(frame=frame, first_month=label)
        out = run347.train_and_predict(
            hindcast=frame.filter(pl.Series(finished)),
            summary_h=summary[finished],
            rows=np.ones(int(finished.sum()), dtype=bool),
            cutoff=stop_start,
            valid_limit=window_end,
            kind="full",
            farm_columns=True,
            targets={"later": (frame.filter(pl.Series(target)), summary[target])},
            seed=seed,
        )
        save(name=name, array=out["later"])
        print(
            f"{name} done {time.time() - started:.0f} s; latest valid time used {latest}",
            flush=True,
        )


def main() -> None:
    """Train the monthly heads of one seed."""
    train(seed=int(sys.argv[sys.argv.index("--seed") + 1]))
    print("all heads done", flush=True)


if __name__ == "__main__":
    main()
