"""Retrain experiment 201's network head (its arm 2) at each month start of the scored year.

For a retrain date `c`, the training and validation pool is every row whose run origin plus 168
hours (the longest lead) is before `c`, so no training target is dated on or after `c`. Every
fourth month of the pool (by the month of the run's origin) is validation, as in experiment 201.
The head is fitted with three seeds and predicts every row whose origin is on or after `c`, so a
quarterly head uses its fit for three months and a monthly head for one. The head trained once is
experiment 201's stored forecasts. Output: `work/head/cut<YYYY-MM>_seed<s>.parquet`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00278-*/e00278_head_run.py [--cuts 2025-07 ...]
"""

import argparse
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Final

import e00278_h_common as common
import e00278_h_data as data
import e00278_h_nn as nn_
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARM: Final[int] = 2
MAX_LEAD: Final[timedelta] = timedelta(hours=168)
CUT_MONTHS: Final[tuple[str, ...]] = tuple(
    f"{y}-{m:02d}" for y, m in [(2025, k) for k in range(7, 13)] + [(2026, k) for k in range(1, 7)]
)
KEYS: Final[tuple[str, ...]] = (
    "site",
    "run",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    "actual",
)


def month_start(*, label: str) -> datetime:
    """Return 00:00 UTC on the first day of a `YYYY-MM` month."""
    year, month = (int(part) for part in label.split("-"))
    return datetime(year, month, 1, tzinfo=UTC)


def split_for_cut(*, frame: pl.DataFrame, cut: datetime) -> np.ndarray:
    """Return `train`, `val`, `test`, or `skip` per row for a retrain at `cut`."""
    origin = frame["power_fcst_init_time"]
    pool = ((origin + MAX_LEAD) < cut).to_numpy()
    test = (origin >= cut).to_numpy()
    months = sorted(frame.filter(pl.Series(pool))["month_key"].unique().to_list())
    val_months = {
        m for i, m in enumerate(months) if i % data.VAL_MONTH_STRIDE == data.VAL_MONTH_STRIDE - 1
    }
    is_val = frame["month_key"].is_in(list(val_months)).to_numpy()
    out = np.full(frame.height, "skip", dtype=object)
    out[pool & ~is_val] = "train"
    out[pool & is_val] = "val"
    out[test] = "test"
    train_valid = frame["valid_time"].filter(pl.Series(pool)).max()
    assert isinstance(train_valid, datetime), "the training pool is empty"
    assert train_valid < cut, "a training target reaches the retrain date"
    return out.astype(str)


def main() -> None:
    """Fit the head at each requested month start with three seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", default=list(CUT_MONTHS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    frame = data.load_rows()
    matrix = data.design_columns(frame=frame, arm=ARM)
    q = nn_.quantile_matrix(frame=frame)
    actual = frame["actual"].to_numpy().astype(np.float64)
    out_dir = common.WORK_DIR / "head"
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in args.cuts:
        split = split_for_cut(frame=frame, cut=month_start(label=label))
        test = split == "test"
        keys = frame.select(list(KEYS)).filter(pl.Series(test))
        for seed in SEEDS:
            target = out_dir / f"cut{label}_seed{seed}.parquet"
            if target.exists():
                continue
            started = time.monotonic()
            fit = nn_.fit_predict(x=matrix, q=q, actual=actual, split=split, seed=seed)
            part = keys.with_columns(
                pl.Series("delta", fit.delta[test]),
                pl.Series("log_scale", fit.log_scale[test]),
                n_train=pl.lit(int((split == "train").sum())),
                epochs=pl.lit(fit.epochs),
                seconds=pl.lit(time.monotonic() - started),
            )
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            part.write_parquet(temporary)
            temporary.rename(target)
            print(label, seed, fit.epochs, f"{time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
