# ruff: noqa: E501
"""Fit the mixture-density head arms of experiment 318 on experiment 310's bands, settings, and refits.

Each arm is experiment 310's `MixtureHead` (three Gaussians, 40 epochs, validation on the last 61
days before the cutoff) with a different input list. Per-band arms train on one band's rows; the
`_pooled` arms train once on the rows of all six bands for a (setting, farm, refit) and are scored
band by band. The rows, plant scale, ceiling, and refit cutoffs come from experiment 310's
`Setting`, so every arm sees the rows experiment 310's arms saw.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00318-*/e00318_fit.py [band|setting|farm|YYYY-MM ...]
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
import e00318_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.tail_calibration import level_column

N_PROCESSES: Final[int] = 4
SHUFFLE_SEED_OFFSET: Final[int] = 1_000


def add_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the lead in whole days, the sine and cosine pairs of the clock, to a Setting frame."""
    return frame.with_columns(
        lead_day=(pl.col("lead_hours") / 24.0).ceil(),
        hour_sin=(2 * np.pi * pl.col("hour") / 24.0).sin(),
        hour_cos=(2 * np.pi * pl.col("hour") / 24.0).cos(),
        doy_sin=(2 * np.pi * pl.col("doy") / 366.0).sin(),
        doy_cos=(2 * np.pi * pl.col("doy") / 366.0).cos(),
    )


def shuffled(*, frame: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return the frame with each added column permuted independently across rows."""
    return frame.with_columns(
        pl.Series(c, frame[c].to_numpy()[rng.permutation(frame.height)])
        for c in (*common.ADDED_COLUMNS, "lead_day")
    )


def make_batch(
    *,
    frame: pl.DataFrame,
    inputs: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    plant_base: bool,
    rows: np.ndarray | None,
) -> dict[str, torch.Tensor]:
    """Return the standardised inputs, base forecast, and target of the chosen rows."""
    pick = np.ones(frame.height, dtype=bool) if rows is None else rows
    x = frame.select(inputs).to_numpy().astype(np.float64)
    base = frame["mu"].to_numpy()[pick] if plant_base else np.zeros(int(pick.sum()))
    return {
        "x": torch.tensor(((x - mean) / std)[pick], dtype=torch.float32),
        "mu": torch.tensor(base, dtype=torch.float32),
        "actual": torch.tensor(
            frame["output"].to_numpy().astype(np.float64)[pick], dtype=torch.float32
        ),
    }


def train_head(
    *,
    train: pl.DataFrame,
    tests: dict[str, pl.DataFrame],
    ceilings: dict[str, float],
    inputs: tuple[str, ...],
    plant_base: bool,
    shuffle: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit one mixture head on `train` and return each test frame's (rows, 11) quantiles."""
    torch.manual_seed(seed)
    cols = list(inputs)
    if shuffle:
        rng = np.random.default_rng(SHUFFLE_SEED_OFFSET + seed)
        train = shuffled(frame=train, rng=rng)
        tests = {k: shuffled(frame=v, rng=rng) for k, v in tests.items()}
    train_x = train.select(cols).to_numpy().astype(np.float64)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    is_val = train["is_val"].to_numpy()
    fit_b = make_batch(frame=train, inputs=cols, mean=mean, std=std, plant_base=plant_base, rows=~is_val)
    val_b = make_batch(frame=train, inputs=cols, mean=mean, std=std, plant_base=plant_base, rows=is_val)
    model = f310.MixtureHead(len(cols))
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
    return {
        key: f310.mixture_quantiles(
            model=model,
            batch=make_batch(
                frame=frame, inputs=cols, mean=mean, std=std, plant_base=plant_base, rows=None
            ),
            ceiling=ceilings[key],
        )
        for key, frame in tests.items()
    }


def out_path(*, band: str, setting: str, site: str, arm: str, seed: int, start: datetime) -> Path:
    """Return a job's output file."""
    return common.PRED_DIR / band / setting / f"{arm}_s{seed}_{site}_{start:%Y-%m}.parquet"


def write(
    *, st: f310.Setting, band: str, setting: str, arm: str, seed: int, start: datetime, q: np.ndarray
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
        **{level_column(level=lv): pl.Series(q[:, i]) for i, lv in enumerate(c310.LEVELS)},
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def new_setting(*, band: str, setting: str, site: str, start: datetime) -> f310.Setting:
    """Return experiment 310's Setting with the added columns on its train and test frames."""
    st = f310.Setting(
        band=band, setting=setting, site=site, start=start, n_spread_bins=f310.N_SPREAD_BINS
    )
    st.train = add_columns(frame=st.train)
    st.test = add_columns(frame=st.test)
    return st


def run_band_job(task: tuple[str, str, str, datetime]) -> str:
    """Run the per-band arms for one (band, setting, farm, refit), skipping outputs that exist."""
    torch.set_num_threads(1)
    band, setting, site, start = task
    started = time.time()
    own = setting == "own_history"
    arms = [a for a in common.PER_BAND_ARMS if own or a not in common.OWN_ONLY_ARMS]
    seeds = common.OWN_SEEDS if own else common.ZERO_SEEDS
    todo = [
        (a, s)
        for a in arms
        for s in seeds
        if not out_path(band=band, setting=setting, site=site, arm=a, seed=s, start=start).exists()
    ]
    if not todo:
        return f"{band} {setting} {site} {start:%Y-%m} skipped"
    st = new_setting(band=band, setting=setting, site=site, start=start)
    for arm, seed in todo:
        inputs, plant_base, shuffle = common.PER_BAND_ARMS[arm]
        q = train_head(
            train=st.train,
            tests={band: st.test},
            ceilings={band: st.ceiling},
            inputs=inputs,
            plant_base=plant_base,
            shuffle=shuffle,
            seed=seed,
        )[band]
        write(st=st, band=band, setting=setting, arm=arm, seed=seed, start=start, q=q)
    return f"{band} {setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def run_pooled_job(task: tuple[str, str, datetime]) -> str:
    """Run the pooled arms for one (setting, farm, refit): one head on all bands, scored per band."""
    torch.set_num_threads(1)
    setting, site, start = task
    started = time.time()
    todo = [
        (a, s)
        for a in common.POOLED_ARMS
        for s in common.OWN_SEEDS
        if not all(
            out_path(band=b, setting=setting, site=site, arm=a, seed=s, start=start).exists()
            for b in c310.BAND_ORDER
        )
    ]
    if not todo:
        return f"pooled {setting} {site} {start:%Y-%m} skipped"
    sts = {
        b: new_setting(band=b, setting=setting, site=site, start=start) for b in c310.BAND_ORDER
    }
    train = pl.concat([st.train for st in sts.values()])
    tests = {b: st.test for b, st in sts.items()}
    ceilings = {b: st.ceiling for b, st in sts.items()}
    for arm, seed in todo:
        inputs, plant_base, shuffle = common.POOLED_ARMS[arm]
        quantiles = train_head(
            train=train,
            tests=tests,
            ceilings=ceilings,
            inputs=inputs,
            plant_base=plant_base,
            shuffle=shuffle,
            seed=seed,
        )
        for b, st in sts.items():
            write(st=st, band=b, setting=setting, arm=arm, seed=seed, start=start, q=quantiles[b])
    return f"pooled {setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def run_any(task: tuple) -> str:
    """Dispatch a band job or a pooled job."""
    return run_band_job(task) if len(task) == 4 else run_pooled_job(task)


def main() -> None:
    """Run all jobs in parallel.

    Arguments restrict the jobs: a band, a setting, a farm, or a refit month keeps matching
    per-band jobs (each kind of argument is a filter); `pooled` or `bands` keeps only that job kind.
    """
    args = set(sys.argv[1:])
    kinds = {a for a in args if a in ("pooled", "bands")}
    keep = args - kinds
    band_tasks = list(
        itertools.product(reversed(c310.BAND_ORDER), c310.SETTINGS, c310.FARMS, f310.REFIT_STARTS)
    )
    pooled_tasks = list(itertools.product(["own_history"], c310.FARMS, f310.REFIT_STARTS))

    def matches(labels: tuple[str, ...], groups: tuple[set[str], ...]) -> bool:
        return all(not (keep & g) or bool(set(labels) & keep & g) for g in groups)

    bands, settings = set(c310.BAND_ORDER), set(c310.SETTINGS)
    farms, months = set(c310.FARMS), {f"{s:%Y-%m}" for s in f310.REFIT_STARTS}
    groups = (bands, settings, farms, months)
    tasks: list[tuple] = []
    if "bands" not in kinds or "pooled" in kinds:
        tasks += [
            t
            for t in pooled_tasks
            if (not (keep & bands)) and matches((t[0], t[1], f"{t[2]:%Y-%m}"), groups)
        ]
    if "pooled" not in kinds or "bands" in kinds:
        tasks += [
            t
            for t in band_tasks
            if matches((t[0], t[1], t[2], f"{t[3]:%Y-%m}"), groups)
        ]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_any, tasks):
            print("finished", done, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
