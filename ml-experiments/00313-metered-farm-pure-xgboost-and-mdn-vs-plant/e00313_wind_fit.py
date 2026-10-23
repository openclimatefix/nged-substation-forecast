# ruff: noqa: E501
"""Fit the pure wind arms of experiment 313 for W1 to W3, in own history and zero history.

`pure_xgb` is XGBoost (depth 6, 500 rounds, 11 quantiles, no base margin) and `pure_mdn` is a
three-component Gaussian mixture on the target (five seeds), both on the six wind columns of
experiment 276's XGBoost plus the 2 m temperature, the hour, and the day of year. Training rows are
ERA5 weather before 2025-06-17 (all rows). Own history trains on the farm's rows and zero history on
the other two metered farms' rows pooled. Each model is applied with ERA5 as the weather (a ceiling,
because ERA5 arrives about 5 days late) and to each of the 51 ensemble members, whose quantiles are
mixed with equal weights as in experiment 285.

Output: `wind/<arm>__<setting>__<farm>__s<seed>__<era5|ens>.npy`, shape (cases, 11).

    PYTHONPATH=ml-experiments/00285-wind-effective-wind-adapters:ml-experiments/00310-metered-farm-long-lead-scoring \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00313-*/e00313_wind_fit.py
"""

import multiprocessing
import os
import time
from functools import partial
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00285_common as c285  # noqa: E402  # ty: ignore[unresolved-import]
import e00285_run as run285  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # ty: ignore[unresolved-import]
import e00313_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402

FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
XGB_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
COLUMNS: Final[list[str]] = [*c285.XGBOOST_FEATURES, "temperature_c", "hour", "doy"]
XGB_ROUNDS: Final[int] = 500
VALIDATION_DAYS: Final[int] = 60
CHUNK: Final[int] = 200_000


def stored(*, arm: str, setting: str, farm: str, seed: int, weather: str) -> Path:
    """Return a prediction file."""
    return common.WIND_DIR / f"{arm}__{setting}__{farm}__s{seed}__{weather}.npy"


def fit_xgb(*, train: pl.DataFrame, seed: int) -> xgb.Booster:
    """Fit the pure XGBoost quantile model (no base margin)."""
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": c285.LEVELS,
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "tree_method": "hist",
        "nthread": 2,
        "seed": seed,
    }
    return xgb.train(
        params,
        xgb.DMatrix(train.select(COLUMNS).to_numpy(), label=train["target"].to_numpy()),
        num_boost_round=XGB_ROUNDS,
    )


def predict_xgb(*, booster: xgb.Booster, frame: pl.DataFrame) -> np.ndarray:
    """Return sorted, clipped quantiles for each row of the frame."""
    out = np.empty((frame.height, 11), dtype=np.float32)
    for start in range(0, frame.height, CHUNK):
        part = frame.slice(start, CHUNK).select(COLUMNS).to_numpy()
        out[start : start + CHUNK] = np.sort(
            np.clip(booster.predict(xgb.DMatrix(part)), 0.0, 1.0), axis=1
        )
    return out


def fit_mdn(*, train: pl.DataFrame, seed: int) -> tuple[f310.MixtureHead, np.ndarray, np.ndarray]:
    """Fit one mixture on the target; return the network and the input means and deviations."""
    torch.manual_seed(seed)
    x = train.select(COLUMNS).to_numpy().astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    cutoff = train.select(pl.col("valid_time").max() - pl.duration(days=VALIDATION_DAYS)).item()
    is_val = (train["valid_time"] >= cutoff).to_numpy()

    def tensors(rows: np.ndarray) -> dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(((x - mean) / std)[rows], dtype=torch.float32),
            "mu": torch.zeros(int(rows.sum()), dtype=torch.float32),
            "actual": torch.tensor(train["target"].to_numpy()[rows], dtype=torch.float32),
        }

    fit_b, val_b = tensors(~is_val), tensors(is_val)
    model = f310.MixtureHead(len(COLUMNS))
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
    return model, mean, std


def predict_mdn(
    *, model: f310.MixtureHead, mean: np.ndarray, std: np.ndarray, frame: pl.DataFrame
) -> np.ndarray:
    """Return the mixture's 11 quantiles for each row, clamped to 0 to 1."""
    out = np.empty((frame.height, 11), dtype=np.float32)
    for start in range(0, frame.height, CHUNK):
        part = frame.slice(start, CHUNK).select(COLUMNS).to_numpy().astype(np.float64)
        batch = {
            "x": torch.tensor((part - mean) / std, dtype=torch.float32),
            "mu": torch.zeros(part.shape[0], dtype=torch.float32),
        }
        out[start : start + CHUNK] = f310.mixture_quantiles(model=model, batch=batch, ceiling=1.0)
    return out


def save(*, path: Path, array: np.ndarray) -> None:
    """Write an array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def worker(farm: str) -> None:
    """Fit and store every pure arm for one farm in both settings."""
    torch.set_num_threads(1)
    train_own, era5_cases, _, members, _ = run285.farm_frames(farm=farm, need_ens=True)
    others = [f for f in FARMS if f != farm]
    train_zero = pl.concat([run285.farm_frames(farm=f, need_ens=False)[0] for f in others])
    for setting, train in (("own_history", train_own), ("zero_history", train_zero)):
        for arm, seeds in (("pure_xgb", XGB_SEEDS), ("pure_mdn", SEEDS)):
            for seed in seeds:
                paths = {
                    w: stored(arm=arm, setting=setting, farm=farm, seed=seed, weather=w)
                    for w in ("era5", "ens")
                }
                if all(p.exists() for p in paths.values()):
                    continue
                started = time.time()
                if arm == "pure_xgb":
                    predict = partial(predict_xgb, booster=fit_xgb(train=train, seed=seed))
                else:
                    net, mean, std = fit_mdn(train=train, seed=seed)
                    predict = partial(predict_mdn, model=net, mean=mean, std=std)

                save(path=paths["era5"], array=predict(frame=era5_cases))
                assert members is not None
                mixed = c285.mixture_quantiles(
                    member_quantiles=predict(frame=members).reshape(-1, c285.MEMBERS, 11)
                )
                save(path=paths["ens"], array=mixed)
                print(
                    f"{farm} {setting} {arm} seed {seed}: {time.time() - started:.0f} s", flush=True
                )


def main() -> None:
    """Start one worker process per farm."""
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, kwargs={"farm": f}) for f in FARMS]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    if any(p.exitcode for p in processes):
        msg = "a worker failed"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
