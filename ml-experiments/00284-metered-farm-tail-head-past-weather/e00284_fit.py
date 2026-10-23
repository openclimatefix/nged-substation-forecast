# ruff: noqa: E501, D107
"""Fit every arm of experiment 284 and write each job's scored-year tail quantiles.

Arms: p2 (the spread-binned plant), t0 (tail head on ensemble rows only), t3 (past-weather plus
ensemble rows with the era flag), t4a (t3 without the flag), t4b (t3 with the past-weather rows'
power shuffled within farm, month, and half-hour), and t5 (XGBoost on the six tail levels). Setting:
own history, one fit on rows before 2025-09. Each job (farm, arm, seed) writes its file atomically
and is skipped when it exists. Run:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00284-*/e00284_fit.py
"""

import itertools
import multiprocessing
import sys
import time
from functools import cache
from typing import Final

import e00284_common as common
import e00284_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from studies.tail_calibration import level_column

N_PROCESSES: Final[int] = 4
EPOCHS: Final[int] = 60
BATCH: Final[int] = 512
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
CROSSING_WEIGHT: Final[float] = 1.0
SHUFFLE_SEED: Final[int] = 284
TAIL_POSITIONS: Final[list[int]] = common.table_columns(levels=common.TAIL_LEVELS)


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


def lag_residual(
    *,
    time_us: np.ndarray,
    lead: np.ndarray,
    lag_power: np.ndarray,
    lag_missing: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Return observed power one day earlier minus the plant forecast then (0 when unobserved)."""
    lag_t = time_us - np.where(lead <= 24, 1, 2) * 86_400_000_000
    idx = np.minimum(np.searchsorted(time_us, lag_t), len(time_us) - 1)
    found = (time_us[idx] == lag_t) & (lag_missing == 0)
    return np.where(found, lag_power - mu[idx], 0.0)


def arrays(*, f: common.Farm) -> dict[str, np.ndarray]:
    """Return a farm's per-row arrays (all rows, both eras) at the farm's own plant scale and ceiling."""
    frame = f.frame
    mu = common.plant_forecast(
        unit=frame["unit_mean"].to_numpy(), scale=f.own_scale, ceiling=f.own_ceiling
    )
    lagres = np.zeros(len(mu))
    time_us = frame["time"].dt.epoch("us").to_numpy()
    era = frame["era"].to_numpy()
    for flag in (0.0, 1.0):
        rows = np.flatnonzero(era == flag)
        lagres[rows] = lag_residual(
            time_us=time_us[rows],
            lead=frame["lead_hours"].to_numpy()[rows],
            lag_power=frame["lag_power"].to_numpy()[rows],
            lag_missing=frame["lag_missing"].to_numpy()[rows],
            mu=mu[rows],
        )
    ghi = frame["ghi_mean"].to_numpy()
    std = frame["ghi_std"].to_numpy()
    head = np.column_stack(
        [
            mu,
            std / (ghi + 1.0),
            std,
            lagres,
            frame["lag_missing"].to_numpy(),
            frame["elevation_deg"].to_numpy(),
            frame["csi"].to_numpy(),
            frame["lead_hours"].to_numpy(),
            frame["temperature_mean"].to_numpy(),
        ]
    )
    return {
        "mu": mu,
        "spread": std,
        "head": np.column_stack([head, era]),
        "output": frame["output"].to_numpy().astype(np.float64),
    }


def shuffled_output(*, f: common.Farm, mask: np.ndarray) -> np.ndarray:
    """Return the farm's output with the masked rows' values permuted within month and half-hour."""
    frame = f.frame
    output = frame["output"].to_numpy().astype(np.float64).copy()
    key = (
        frame["time"].dt.strftime("%Y-%m").to_numpy().astype(str)
        + "_"
        + (frame["time"].dt.hour() * 2 + frame["time"].dt.minute() // 30)
        .cast(pl.String)
        .to_numpy()
        .astype(str)
    )
    rng = np.random.default_rng(SHUFFLE_SEED)
    for k in np.unique(key[mask]):
        rows = np.flatnonzero(mask & (key == k))
        output[rows] = output[rng.permutation(rows)]
    return output


class Setting:
    """The training pools, the spread-binned plant's table, and the scored rows of one farm."""

    def __init__(self, *, site: str) -> None:
        self.site = site
        f = farm(site)
        a = arrays(f=f)
        self.ceiling = f.own_ceiling
        shuffled = shuffled_output(f=f, mask=f.past_train)
        self.kinds: dict[str, dict[str, np.ndarray]] = {}
        for kind, mask in (
            ("ens_train", f.ens_train),
            ("ens_val", f.ens_val),
            ("past_train", f.past_train),
        ):
            self.kinds[kind] = {k: v[mask] for k, v in a.items()}
        self.kinds["placebo"] = {
            **self.kinds["past_train"],
            "output": shuffled[f.past_train],
        }
        tuning = self.pool("ens_train", "ens_val")
        self.edges = common.spread_bin_table(
            mu=tuning["mu"], spread=tuning["spread"], output=tuning["output"]
        )
        self.test = {k: v[f.scored] for k, v in a.items()}
        self.test_frame = f.frame.filter(pl.Series(f.scored))

    def pool(self, *names: str) -> dict[str, np.ndarray]:
        """Return the concatenation of the named training kinds."""
        return {k: np.concatenate([self.kinds[n][k] for n in names]) for k in self.kinds[names[0]]}

    def offsets(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Return the (rows, 6) spread-binned offsets at the tail levels."""
        full = common.spread_offsets_for(mu=data["mu"], spread=data["spread"], edges=self.edges)
        return full[:, TAIL_POSITIONS]


def make_batch(
    *, data: dict[str, np.ndarray], st: Setting, stats: tuple, device: str
) -> dict[str, torch.Tensor]:
    """Return a split's tensors: standardised head inputs, plant forecast, offsets, and targets."""
    mean, std = stats

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)

    return {
        "head": tensor((data["head"] - mean) / std),
        "mu": tensor(data["mu"]),
        "offsets": tensor(st.offsets(data)),
        "ceiling": torch.full((len(data["mu"]),), st.ceiling, dtype=torch.float32, device=device),
        "actual": tensor(data["output"]),
    }


def pinball(*, q: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over rows and the tail levels."""
    error = actual[:, None] - q
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def train_tail_head(*, arm: str, st: Setting, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one tail head and return the scored rows' (rows, 6) quantiles and a per-row crossing flag."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    names = {
        "t0": ("ens_train",),
        "t3": ("ens_train", "past_train"),
        "t4a": ("ens_train", "past_train"),
        "t4b": ("ens_train", "placebo"),
    }[arm]
    use_era = arm != "t4a"

    def view(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return data if use_era else {**data, "head": data["head"][:, :-1]}

    train, val, test = view(st.pool(*names)), view(st.kinds["ens_val"]), view(st.test)
    stats = (train["head"].mean(axis=0), train["head"].std(axis=0) + 1e-6)
    levels = torch.tensor(common.TAIL_LEVELS, dtype=torch.float32, device=device)
    train_b = make_batch(data=train, st=st, stats=stats, device=device)
    val_b = make_batch(data=val, st=st, stats=stats, device=device)
    test_b = make_batch(data=test, st=st, stats=stats, device=device)
    model = models.TailHead(train["head"].shape[1]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(train["mu"])

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            return float(pinball(q=model(val_b), actual=val_b["actual"], levels=levels))

    best, best_state = float("inf"), {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, BATCH):
            batch = {k: v[order[start : start + BATCH]] for k, v in train_b.items()}
            loss = pinball(q=model(batch), actual=batch["actual"], levels=levels)
            loss = loss + CROSSING_WEIGHT * model.crossing_penalty(batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        score = validate()
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return (
            model(test_b).cpu().numpy().astype(np.float64),
            model.crossed(test_b).cpu().numpy(),
        )


def tail_xgboost(*, st: Setting) -> tuple[np.ndarray, np.ndarray]:
    """Fit XGBoost on the six tail levels with the tail head's inputs (t5)."""
    train = st.pool("ens_train", "ens_val", "past_train")
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=common.TAIL_LEVELS,
        tree_method="hist",
        n_estimators=150,
        learning_rate=0.15,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        n_jobs=1,
        random_state=284,
    )
    model.fit(train["head"], train["output"])
    predicted = np.asarray(model.predict(st.test["head"])).reshape(len(st.test["mu"]), -1)
    crossed = (np.diff(predicted, axis=1) < 0).any(axis=1)
    return np.maximum(np.sort(predicted, axis=1), 0.0), crossed


def spread_binned_plant(*, st: Setting) -> tuple[np.ndarray, np.ndarray]:
    """Return the spread-binned plant's tail quantiles on the scored rows (p2)."""
    q = st.test["mu"][:, None] + st.offsets(st.test)
    q = np.minimum(np.maximum(np.sort(q, axis=1), 0.0), st.ceiling)
    return q, np.zeros(len(q), dtype=bool)


def out_path(*, site: str, arm: str, seed: int):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / f"{arm}_s{seed}_{site}.parquet"


def write(*, st: Setting, arm: str, seed: int, quantiles: np.ndarray, crossed: np.ndarray) -> None:
    """Write a job's scored rows atomically."""
    path = out_path(site=st.site, arm=arm, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test_frame.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        actual=pl.Series(st.test["output"]),
        crossed=pl.Series(crossed),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i])
            for i, lv in enumerate(common.TAIL_LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_group(site: str) -> str:
    """Run every arm and seed for one farm, skipping jobs whose file exists."""
    torch.set_num_threads(1)
    st = Setting(site=site)
    started = time.time()
    jobs = [("p2", 0), ("t5", 0)]
    jobs += list(itertools.product(common.NETWORK_ARMS, common.SEEDS))
    for arm, seed in jobs:
        if out_path(site=site, arm=arm, seed=seed).exists():
            continue
        if arm == "p2":
            q, crossed = spread_binned_plant(st=st)
        elif arm == "t5":
            q, crossed = tail_xgboost(st=st)
        else:
            q, crossed = train_tail_head(arm=arm, st=st, seed=seed)
        write(st=st, arm=arm, seed=seed, quantiles=q, crossed=crossed)
        print(f"{site} {arm} s{seed} done {time.time() - started:.0f}s", flush=True)
    return site


def main() -> None:
    """Run the farms in parallel; arguments name the farms to keep."""
    farms = [s for s in common.FARMS if not sys.argv[1:] or s in sys.argv[1:]]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, farms):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
