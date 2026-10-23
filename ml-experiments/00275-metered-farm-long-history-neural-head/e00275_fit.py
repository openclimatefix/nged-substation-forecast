# ruff: noqa: E501, D107
"""Fit every arm of experiment 275 and write each job's scored-year quantiles.

Arms: l0 (ensemble rows only), l1 (past-weather plus ensemble rows), l2 (l1 plus the era flag), l3
(l2 pretrained on past weather, fine-tuned on ensemble rows), control (l2 with the past-weather
rows' power shuffled within farm, month, and half-hour), and XGBoost x0 and x2. Settings: `own_history`
trains on the scored farm only; `zero_history` trains on the other five farms and uses the median
plant scale and ceiling of those five. Each job (setting, farm, arm, seed) writes its file atomically
and is skipped when it exists, so a crashed run resumes. Run:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00275-*/e00275_fit.py own_history
"""

import itertools
import multiprocessing
import sys
import time
from functools import cache

import e00275_common as common
import e00275_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from studies.tail_calibration import level_column

N_PROCESSES = 4
EPOCHS = 60
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
BATCH = 512
LEARNING_RATE = 2e-3
FINETUNE_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SHUFFLE_SEED = 275


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
    """Return observed power one day earlier minus the plant forecast then (0 when unobserved).

    The rows must be sorted by time and hold one row per valid time.
    """
    lag_t = time_us - np.where(lead <= 24, 1, 2) * 86_400_000_000
    idx = np.minimum(np.searchsorted(time_us, lag_t), len(time_us) - 1)
    found = (time_us[idx] == lag_t) & (lag_missing == 0)
    return np.where(found, lag_power - mu[idx], 0.0)


def arrays(*, f: common.Farm, scale: float, ceiling: float) -> dict[str, np.ndarray]:
    """Return a farm's per-row arrays (all rows, both eras) at the setting's plant scale and ceiling."""
    frame = f.frame
    mu = common.plant_forecast(unit=frame["unit_mean"].to_numpy(), scale=scale, ceiling=ceiling)
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
    raw = frame.select(list(common.RAW_FEATURES)).to_numpy().astype(np.float64)
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
        "raw": np.column_stack([raw, era]),
        "head": np.column_stack([head, era]),
        "output": frame["output"].to_numpy().astype(np.float64),
        "unit": frame["unit_mean"].to_numpy(),
        "time_us": time_us,
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
    """The training pool and the scored rows of one (setting, scored farm)."""

    def __init__(self, *, setting: str, site: str) -> None:
        self.site = site
        others = [s for s in common.FARMS if s != site]
        train_sites = [site] if setting == "own_history" else others
        if setting == "own_history":
            self.scale, self.ceiling = farm(site).own_scale, farm(site).own_ceiling
        else:
            self.scale = float(np.median([farm(s).own_scale for s in others]))
            self.ceiling = float(np.median([farm(s).own_ceiling for s in others]))
        kinds = {"ens_train": [], "ens_val": [], "past_train": [], "control": []}
        for s in train_sites:
            f = farm(s)
            a = arrays(f=f, scale=self.scale, ceiling=self.ceiling)
            shuffled = shuffled_output(f=f, mask=f.past_train)
            for kind, mask in (
                ("ens_train", f.ens_train),
                ("ens_val", f.ens_val),
                ("past_train", f.past_train),
                ("control", f.past_train),
            ):
                piece = {k: v[mask] for k, v in a.items()}
                if kind == "control":
                    piece["output"] = shuffled[mask]
                kinds[kind].append(piece)
        self.kinds = {
            kind: {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            for kind, pieces in kinds.items()
        }
        test = farm(site)
        a = arrays(f=test, scale=self.scale, ceiling=self.ceiling)
        self.test = {k: v[test.scored] for k, v in a.items()}
        self.test_frame = test.frame.filter(pl.Series(test.scored))

    def pool(self, *names: str) -> dict[str, np.ndarray]:
        """Return the concatenation of the named training kinds."""
        return {k: np.concatenate([self.kinds[n][k] for n in names]) for k in self.kinds[names[0]]}


def drop_era(*, data: dict[str, np.ndarray], use_era: bool) -> dict[str, np.ndarray]:
    """Return the data with the era column removed from the input matrices when the arm lacks the flag."""
    if use_era:
        return data
    return {**data, "raw": data["raw"][:, :-1], "head": data["head"][:, :-1]}


def make_batch(
    *, data: dict[str, np.ndarray], stats: tuple, offsets: tuple, ceiling: float, device: str
) -> dict[str, torch.Tensor]:
    """Return a split's tensors: standardised head inputs, plant forecast, offsets, and targets."""
    mean, std = stats
    n = len(data["mu"])

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)

    return {
        "head": tensor((data["head"] - mean) / std),
        "mu": tensor(data["mu"]),
        "offsets": tensor(common.offsets_for(mu=data["mu"], edges=offsets[0], table=offsets[1])),
        "ceiling": torch.full((n,), ceiling, dtype=torch.float32, device=device),
        "actual": tensor(data["output"]),
    }


def pinball(*, q: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over rows and the 11 levels."""
    error = actual[:, None] - q
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def run_epochs(
    *,
    model: models.QuantileHead,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    levels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    generator: torch.Generator,
    keep_start: bool,
) -> None:
    """Train in place and restore the weights of the epoch with the lowest validation loss."""
    device = train["mu"].device
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    n = len(train["mu"])

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            return float(pinball(q=model(val), actual=val["actual"], levels=levels))

    best = validate() if keep_start else float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            batch = {k: v[idx] for k, v in train.items()}
            loss = pinball(q=model(batch), actual=batch["actual"], levels=levels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        score = validate()
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)


def train_neural(*, arm: str, st: Setting, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one network and return the scored farm's (rows, 11) quantiles and a per-row crossing flag."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    use_era = arm in ("l2", "l3", "control", *common.POSTHOC_ARMS)
    train_names = {
        "l0": ("ens_train",),
        "l1": ("ens_train", "past_train"),
        "l2": ("ens_train", "past_train"),
        "l3": ("ens_train", "past_train"),
        "control": ("ens_train", "control"),
        "l2_ens_offsets": ("ens_train", "past_train"),
        "l2_pre_ens": ("ens_train", "past_train"),
    }[arm]
    pooled = st.pool(*train_names)
    if arm == "l2_pre_ens":
        ens_start_us = int(common.ENS_START.timestamp() * 1_000_000)
        keep = (pooled["raw"][:, -1] == 0.0) | (pooled["time_us"] < ens_start_us)
        pooled = {k: v[keep] for k, v in pooled.items()}
    train = drop_era(data=pooled, use_era=use_era)
    val = drop_era(data=st.kinds["ens_val"], use_era=use_era)
    test = drop_era(data=st.test, use_era=use_era)
    ens = st.pool("ens_train", "ens_val") if arm == "l2_ens_offsets" else None
    offsets_train = train if ens is None else ens
    offsets_val = val if ens is None else {"mu": ens["mu"][:0], "output": ens["output"][:0]}
    offsets = common.bin_offsets(
        mu=np.concatenate([offsets_train["mu"], offsets_val["mu"]]),
        output=np.concatenate([offsets_train["output"], offsets_val["output"]]),
    )
    stats = (train["head"].mean(axis=0), train["head"].std(axis=0) + 1e-6)
    kwargs = {"stats": stats, "offsets": offsets, "ceiling": st.ceiling, "device": device}
    levels = torch.tensor(common.LEVELS, dtype=torch.float32, device=device)
    val_b, test_b = make_batch(data=val, **kwargs), make_batch(data=test, **kwargs)
    model = models.QuantileHead(train["head"].shape[1]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if arm == "l3":
        past_b = make_batch(data=drop_era(data=st.pool("past_train"), use_era=True), **kwargs)
        run_epochs(
            model=model,
            train=past_b,
            val=val_b,
            levels=levels,
            epochs=PRETRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            generator=generator,
            keep_start=False,
        )
        ens_b = make_batch(data=drop_era(data=st.pool("ens_train"), use_era=True), **kwargs)
        run_epochs(
            model=model,
            train=ens_b,
            val=val_b,
            levels=levels,
            epochs=FINETUNE_EPOCHS,
            learning_rate=FINETUNE_LEARNING_RATE,
            generator=generator,
            keep_start=True,
        )
    else:
        train_b = make_batch(data=train, **kwargs)
        run_epochs(
            model=model,
            train=train_b,
            val=val_b,
            levels=levels,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            generator=generator,
            keep_start=False,
        )
    model.eval()
    with torch.no_grad():
        return (
            model(test_b).cpu().numpy().astype(np.float64),
            model.crossed(test_b).cpu().numpy(),
        )


def xgboost_quantiles(*, arm: str, st: Setting) -> tuple[np.ndarray, np.ndarray]:
    """Fit XGBoost with 11 quantile outputs (x0 on ensemble rows, x2 adding past-weather rows and the era flag)."""
    use_era = arm == "x2"

    def matrix(ds: dict[str, np.ndarray]) -> np.ndarray:
        ds = drop_era(data=ds, use_era=use_era)
        return np.column_stack([ds["raw"], ds["mu"]])

    names = ("ens_train", "ens_val", "past_train") if use_era else ("ens_train", "ens_val")
    train = st.pool(*names)
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
        random_state=275,
    )
    model.fit(matrix(train), train["output"])
    predicted = np.asarray(model.predict(matrix(st.test))).reshape(len(st.test["mu"]), -1)
    crossed = (np.diff(predicted, axis=1) < 0).any(axis=1)
    return np.maximum(np.sort(predicted, axis=1), 0.0), crossed


def out_path(*, setting: str, site: str, arm: str, seed: int):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s{seed}_{site}.parquet"


def write(
    *, st: Setting, setting: str, arm: str, seed: int, quantiles: np.ndarray, crossed: np.ndarray
) -> None:
    """Write a job's scored rows atomically."""
    path = out_path(setting=setting, site=st.site, arm=arm, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test_frame.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit(setting),
        actual=pl.Series(st.test["output"]),
        crossed=pl.Series(crossed),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_group(task: tuple[str, str]) -> str:
    """Run every arm and seed for one (setting, farm), skipping jobs whose file exists."""
    torch.set_num_threads(1)
    setting, site = task
    st = Setting(setting=setting, site=site)
    started = time.time()
    jobs = [(a, 0) for a in common.XGB_ARMS]
    jobs += list(itertools.product((*common.NEURAL_ARMS, *common.POSTHOC_ARMS), common.SEEDS))
    if setting == "zero_history":
        jobs = [(a, s) for a, s in jobs if a in ("l2", "x2")]
    for arm, seed in jobs:
        if out_path(setting=setting, site=site, arm=arm, seed=seed).exists():
            continue
        if arm in common.XGB_ARMS:
            q, crossed = xgboost_quantiles(arm=arm, st=st)
        else:
            q, crossed = train_neural(arm=arm, st=st, seed=seed)
        write(st=st, setting=setting, arm=arm, seed=seed, quantiles=q, crossed=crossed)
        print(f"{setting} {site} {arm} s{seed} done {time.time() - started:.0f}s", flush=True)
    return f"{setting} {site}"


def main() -> None:
    """Run the (setting, farm) groups in parallel; arguments name settings or farms to keep."""
    tasks = list(itertools.product(common.SETTINGS, common.FARMS))
    only = sys.argv[1:]
    if only:
        tasks = [t for t in tasks if t[0] in only or t[1] in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, tasks):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
