# ruff: noqa: E501
"""Fit the pure solar arms of experiment 313 on experiment 310's day bands, settings, and refits.

`pure_xgb` is XGBoost (depth 6, 300 rounds, no base margin) and `pure_mdn` is a three-component
Gaussian mixture on the target itself (five seeds), both on weather, sun geometry, and calendar
columns only (`e00313_common.SOLAR_PURE_COLUMNS`). The training and scored rows come from
experiment 310's `Setting`, so the rows match its arms exactly.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00313-*/e00313_solar_fit.py [band|setting|farm|YYYY-MM ...]
"""

import itertools
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00313_common as common
import numpy as np
import polars as pl
import torch
import xgboost as xgb
from studies.tail_calibration import level_column

N_PROCESSES: Final[int] = 4
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
XGB_ROUNDS: Final[int] = 300
if set(common.SOLAR_PURE_COLUMNS) & common.PLANT_DERIVED:
    msg = "plant-derived columns in the pure inputs"
    raise ValueError(msg)


def xgb_quantiles(*, st: f310.Setting, seed: int) -> np.ndarray:
    """Fit XGBoost on the pure columns with no base margin; return sorted, clamped quantiles."""
    columns = list(common.SOLAR_PURE_COLUMNS)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=c310.LEVELS,
        tree_method="hist",
        n_estimators=XGB_ROUNDS,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=1.0,
        n_jobs=1,
        random_state=seed,
    )
    model.fit(st.train.select(columns).to_numpy(), st.train["output"].to_numpy().astype(np.float64))
    raw = np.asarray(model.predict(st.test.select(columns).to_numpy())).reshape(st.test.height, -1)
    return st.clamp(np.sort(raw, axis=1))


def mdn_quantiles(*, st: f310.Setting, seed: int) -> np.ndarray:
    """Fit one mixture on the pure columns, with the plant forecast replaced by zero."""
    torch.manual_seed(seed)
    columns = list(common.SOLAR_PURE_COLUMNS)
    train, test = st.train, st.test
    train_x = train.select(columns).to_numpy().astype(np.float64)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    is_val = train["is_val"].to_numpy()

    def batch(frame: pl.DataFrame, rows: np.ndarray | None) -> dict[str, torch.Tensor]:
        pick = np.ones(frame.height, dtype=bool) if rows is None else rows
        x = frame.select(columns).to_numpy().astype(np.float64)
        return {
            "x": torch.tensor(((x - mean) / std)[pick], dtype=torch.float32),
            "mu": torch.zeros(int(pick.sum()), dtype=torch.float32),
            "actual": torch.tensor(
                frame["output"].to_numpy().astype(np.float64)[pick], dtype=torch.float32
            ),
        }

    fit_b, val_b, test_b = batch(train, ~is_val), batch(train, is_val), batch(test, None)
    model = f310.MixtureHead(len(columns))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=f310.LEARNING_RATE, weight_decay=f310.WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    n = len(fit_b["mu"])
    for _ in range(f310.EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, f310.BATCH):
            idx = order[start : start + f310.BATCH]
            loss = model.nll({k: v[idx] for k, v in fit_b.items()})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.nll(val_b))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return f310.mixture_quantiles(model=model, batch=test_b, ceiling=st.ceiling)


def out_path(*, band: str, setting: str, site: str, arm: str, seed: int, start: datetime) -> Path:
    """Return a job's output file."""
    return common.SOLAR_PRED_DIR / band / setting / f"{arm}_s{seed}_{site}_{start:%Y-%m}.parquet"


def write(
    *,
    st: f310.Setting,
    band: str,
    setting: str,
    arm: str,
    seed: int,
    start: datetime,
    quantiles: np.ndarray,
) -> None:
    """Write a job's test rows atomically."""
    path = out_path(band=band, setting=setting, site=st.site, arm=arm, seed=seed, start=start)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit(setting),
        band=pl.lit(band),
        actual=st.test["output"].cast(pl.Float64),
        **{level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(c310.LEVELS)},
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_job(task: tuple[str, str, str, datetime]) -> str:
    """Run both pure arms for one (band, setting, farm, refit), skipping outputs that exist."""
    torch.set_num_threads(1)
    band, setting, site, start = task
    started = time.time()
    jobs = [("pure_xgb", 0)] + [("pure_mdn", s) for s in SEEDS]
    todo = [
        j
        for j in jobs
        if not out_path(
            band=band, setting=setting, site=site, arm=j[0], seed=j[1], start=start
        ).exists()
    ]
    if not todo:
        return f"{band} {setting} {site} {start:%Y-%m} skipped"
    st = f310.Setting(
        band=band, setting=setting, site=site, start=start, n_spread_bins=f310.N_SPREAD_BINS
    )
    for arm, seed in todo:
        q = (
            xgb_quantiles(st=st, seed=seed)
            if arm == "pure_xgb"
            else mdn_quantiles(st=st, seed=seed)
        )
        write(st=st, band=band, setting=setting, arm=arm, seed=seed, start=start, quantiles=q)
    return f"{band} {setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def main() -> None:
    """Run all jobs in parallel; arguments name bands, settings, farms, or refit months."""
    tasks = list(
        itertools.product(reversed(c310.BAND_ORDER), c310.SETTINGS, c310.FARMS, f310.REFIT_STARTS)
    )
    only = sys.argv[1:]
    if only:
        tasks = [
            t
            for t in tasks
            if t[0] in only or t[1] in only or t[2] in only or f"{t[3]:%Y-%m}" in only
        ]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_job, tasks):
            print("finished", done, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
