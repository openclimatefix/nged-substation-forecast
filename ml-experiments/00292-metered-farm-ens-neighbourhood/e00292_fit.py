# ruff: noqa: D107
"""Fit every arm of experiment 292 (own history) and write each job's scored-year quantiles.

Arms: `n0` (H5 on the own cell), `n1` (H5 plus the 11 neighbourhood summaries), `placebo` (n1 with
summaries from another day), `n2` (H5 plus a convolution over the 19 cells), `x0` and `x1` (XGBoost
without and with the summaries). One job is one (farm, arm, seed); its output file is written
atomically and skipped when it exists. Pass arm names to run only those arms:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00292-*/e00292_fit.py n0 n1 placebo x0 x1
"""

import itertools
import multiprocessing
import os
import sys
import time
from functools import cache

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import e00292_common as common
import e00292_models as models
import numpy as np
import polars as pl
import torch
import xgboost as xgb
from studies.tail_calibration import level_column

N_PROCESSES = 4
EPOCHS = 60
BATCH = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
CSI_BANDS = (0.3, 0.7)
PLACEBO_SEED = 292


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


def lag_residual(*, f: common.Farm, mu: np.ndarray) -> np.ndarray:
    """Return observed power one day earlier minus the plant forecast then (0 when unobserved)."""
    t = f.frame["time"].dt.epoch("us").to_numpy()
    days = np.where(f.frame["lead_hours"].to_numpy() <= 24, 1, 2)
    lag_t = t - days * 86_400_000_000
    idx = np.searchsorted(t, lag_t)
    idx_c = np.minimum(idx, len(t) - 1)
    found = (t[idx_c] == lag_t) & (f.frame["lag_missing"].to_numpy() == 0)
    return np.where(found, f.frame["lag_power"].to_numpy() - mu[idx_c], 0.0)


def placebo_summaries(*, f: common.Farm) -> np.ndarray:
    """Return the summaries of another day, same split, month, hour, and own-cell clear-sky band."""
    frame = f.frame
    summaries = frame.select(list(common.SUMMARY_FEATURES)).to_numpy().astype(np.float64)
    split = np.where(f.tuning, 0, 1)
    band = np.digitize(frame["csi"].to_numpy(), CSI_BANDS)
    keys = (
        frame["time"].dt.month().to_numpy().astype(np.int64) * 1000
        + frame["time"].dt.hour().to_numpy().astype(np.int64) * 10
        + band
    ) * 2 + split
    days = frame["time"].dt.date().cast(pl.Int32).to_numpy()
    rng = np.random.default_rng(PLACEBO_SEED)
    out = summaries.copy()
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        for row in rows:
            others = rows[days[rows] != days[row]]
            if len(others):
                out[row] = summaries[rng.choice(others)]
    return out


def image_for(*, f: common.Farm) -> np.ndarray:
    """Return the (rows, 4, 5, 5) standardised image of the 19 cells aligned to the farm's rows."""
    raw = np.load(common.NEIGHBOUR_DIR / f"{f.site}_image.npy")
    times = np.load(common.NEIGHBOUR_DIR / f"{f.site}_image_time.npy")
    order = np.searchsorted(times, f.frame["time"].dt.epoch("us").to_numpy())
    raw = raw[order]
    mask = raw[:, 2:3]
    centre = raw[:, 0:1, 2:3, 2:3]
    return np.concatenate(
        [raw[:, 0:1] / 1000.0, raw[:, 1:2] / 300.0, mask, mask * (raw[:, 0:1] - centre) / 300.0],
        axis=1,
    ).astype(np.float32)


def features(*, f: common.Farm, scale: float, ceiling: float) -> dict[str, np.ndarray]:
    """Return a farm's raw, head, summary, placebo, and image arrays and the plant forecast."""
    frame = f.frame
    mu = common.plant_forecast(farm=f, scale=scale, ceiling=ceiling)
    raw = frame.select(list(common.RAW_FEATURES)).to_numpy().astype(np.float64)
    ghi = frame["ghi_mean"].to_numpy()
    head = np.column_stack(
        [
            mu,
            frame["ghi_std"].to_numpy() / (ghi + 1.0),
            frame["ghi_std"].to_numpy(),
            lag_residual(f=f, mu=mu),
            frame["lag_missing"].to_numpy(),
            frame["elevation_deg"].to_numpy(),
            frame["csi"].to_numpy(),
            frame["lead_hours"].to_numpy(),
            frame["temperature_mean"].to_numpy(),
        ]
    )
    summ = frame.select(list(common.SUMMARY_FEATURES)).to_numpy().astype(np.float64)
    return {
        "mu": mu,
        "raw": raw,
        "head": head,
        "summ": summ,
        "placebo": placebo_summaries(f=f),
        "image": image_for(f=f),
    }


class Setting:
    """The training, validation, and scored arrays of one farm (own history)."""

    def __init__(self, *, site: str) -> None:
        self.site = site
        f = farm(site)
        self.scale, self.ceiling = f.own_scale, f.own_ceiling
        feats = features(f=f, scale=self.scale, ceiling=self.ceiling)
        output = f.frame["output"].to_numpy().astype(np.float64)
        masks = {
            "train": f.tuning & ~f.validation,
            "val": f.tuning & f.validation,
            "test": f.scored,
        }
        # Rows without neighbourhood summaries cannot be fitted or scored by any arm.
        usable = ~np.isnan(feats["summ"]).any(axis=1)
        masks = {k: m & usable for k, m in masks.items()}
        self.split = {
            key: {"output": output[m], **{k: v[m] for k, v in feats.items()}}
            for key, m in masks.items()
        }
        tuned = np.concatenate([self.split["train"]["mu"], self.split["val"]["mu"]])
        out_all = np.concatenate([self.split["train"]["output"], self.split["val"]["output"]])
        self.edges, self.table = common.bin_offsets(mu=tuned, output=out_all)
        self.test_frame = f.frame.filter(pl.Series(masks["test"]))
        self.stats = {
            name: (
                self.split["train"][name].mean(axis=0),
                self.split["train"][name].std(axis=0) + 1e-6,
            )
            for name in ("raw", "head", "summ", "placebo")
        }

    def inputs(self, *, split: str, arm: str) -> np.ndarray:
        """Return the standardised head matrix of an arm."""
        ds = self.split[split]
        parts = [(ds["head"] - self.stats["head"][0]) / self.stats["head"][1]]
        if arm in ("n1", "n2_summ"):
            parts.append((ds["summ"] - self.stats["summ"][0]) / self.stats["summ"][1])
        if arm == "placebo":
            parts.append((ds["placebo"] - self.stats["summ"][0]) / self.stats["summ"][1])
        return np.column_stack(parts)


def make_batch(*, st: Setting, split: str, arm: str, device: str) -> dict[str, torch.Tensor]:
    """Return a split's tensors for an arm."""
    ds = st.split[split]
    n = len(ds["mu"])
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    batch = {
        "head": to(st.inputs(split=split, arm=arm)),
        "mu": to(ds["mu"]),
        "offsets": to(common.offsets_for(mu=ds["mu"], edges=st.edges, table=st.table)),
        "ceiling": torch.full((n,), st.ceiling, dtype=torch.float32, device=device),
        "actual": to(ds["output"]),
    }
    if arm == "n2":
        batch["image"] = to(ds["image"])
    return batch


def pinball(*, q: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over rows and the 11 levels."""
    error = actual[:, None] - q
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def train_neural(*, arm: str, st: Setting, seed: int) -> np.ndarray:
    """Fit one network and return the scored farm's (rows, 11) quantiles."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    levels = torch.tensor(common.LEVELS, dtype=torch.float32, device=device)
    train, val, test = (
        make_batch(st=st, split=s, arm=arm, device=device) for s in ("train", "val", "test")
    )
    n_in = train["head"].shape[1]
    model = (models.ConvQuantileHead(n_in) if arm == "n2" else models.QuantileHead(n_in)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(train["mu"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            batch = {k: v[idx] for k, v in train.items()}
            loss = pinball(q=model(batch), actual=batch["actual"], levels=levels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(pinball(q=model(val), actual=val["actual"], levels=levels))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(test).cpu().numpy().astype(np.float64)


def xgboost_quantiles(*, st: Setting, with_summaries: bool) -> np.ndarray:
    """Fit XGBoost with 11 quantile outputs on the tuning rows and return the scored quantiles."""

    def matrix(ds: dict[str, np.ndarray]) -> np.ndarray:
        parts = [ds["raw"], ds["mu"][:, None]]
        if with_summaries:
            parts.append(ds["summ"])
        return np.column_stack(parts)

    train_x = np.concatenate([matrix(st.split["train"]), matrix(st.split["val"])])
    train_y = np.concatenate([st.split["train"]["output"], st.split["val"]["output"]])
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=common.LEVELS,
        tree_method="hist",
        n_estimators=150,
        learning_rate=0.15,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        n_jobs=1,
        random_state=292,
    )
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(matrix(st.split["test"]))).reshape(
        len(st.split["test"]["mu"]), -1
    )
    return np.maximum(np.sort(predicted, axis=1), 0.0)


def out_path(*, site: str, arm: str, seed: int):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / "own_history" / f"{arm}_s{seed}_{site}.parquet"


def write(*, st: Setting, arm: str, seed: int, quantiles: np.ndarray) -> None:
    """Write a job's scored rows atomically, with the quantiles sorted so none cross."""
    quantiles = np.sort(quantiles, axis=1)
    path = out_path(site=st.site, arm=arm, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test_frame.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit("own_history"),
        actual=pl.Series(st.split["test"]["output"]),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_farm(task: tuple[str, tuple[str, ...]]) -> str:
    """Run every requested arm and seed for one farm, skipping jobs whose file exists."""
    torch.set_num_threads(1)
    site, arms = task
    st = Setting(site=site)
    started = time.time()
    jobs = [
        (arm, seed)
        for arm in arms
        for seed in (common.SEEDS if arm in common.NEURAL_ARMS else (0,))
    ]
    for arm, seed in jobs:
        if out_path(site=site, arm=arm, seed=seed).exists():
            continue
        if arm in common.XGB_ARMS:
            q = xgboost_quantiles(st=st, with_summaries=arm == "x1")
        else:
            q = train_neural(arm=arm, st=st, seed=seed)
        write(st=st, arm=arm, seed=seed, quantiles=q)
        print(f"{site} {arm} s{seed} done {time.time() - started:.0f}s", flush=True)
    return site


def main() -> None:
    """Run the requested arms for all farms in parallel."""
    arms = tuple(sys.argv[1:]) or (*common.NEURAL_ARMS, *common.XGB_ARMS)
    tasks = list(itertools.product(common.FARMS, [arms]))
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_farm, tasks):
            print("farm finished", done, flush=True)


if __name__ == "__main__":
    main()
