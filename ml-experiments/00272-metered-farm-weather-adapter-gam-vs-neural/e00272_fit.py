# ruff: noqa: E501, D107
"""Fit every arm of experiment 272 in both settings and write each job's scored-year quantiles.

`own_history` fits on the scored farm's own tuning months; `zero_history` fits on the other five
farms' tuning months, with the plant scale and ceiling the median over those five farms. The
adapters correct the plant's irradiance (a factor exp(g)) and cell temperature (an offset h); see
`e00272_models`. A job is one (setting, farm, arm, seed); its file is written atomically and
skipped when it exists. Run, giving the arm group (`phase1` or `phase2`) and optionally farms:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00272-*/e00272_fit.py phase1
"""

import itertools
from collections.abc import Callable
import multiprocessing
import os
import sys
import time
from functools import cache

import e00272_common as common
import e00272_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.tail_calibration import level_column

N_PROCESSES = 4
EPOCHS = int(os.environ.get("E272_EPOCHS", "60"))
BATCH = 512
PENALTIES = {
    "gam": ((1e-3, 1e-4), (1e-2, 1e-4)),
    "nn": ((0.0, 1e-4), (0.0, 1e-2)),
}
"""(second-difference penalty, weight decay) candidates; the choice is made on validation rows."""
LEARNING_RATE = {"gam": 1e-2, "nn": 2e-3}
ALT_EPOCHS = 15
CSI_BANDS = (0.3, 0.6, 0.9)
PHASE1 = ("a2_gam_min", "a3_gam_full")
PHASE2 = (
    "a4_nn_min",
    "a5_nn_full",
    "a3_head",
    "a5_head",
    "a1_head",
    "a3_shuffle",
    "a3_placebo",
    "a5_shuffle",
    "a5_placebo",
    "a3_alt",
    "a5_alt",
    "blind",
)
ARM_SEEDS = {
    "a2_gam_min": common.GAM_SEEDS,
    "a3_gam_full": common.GAM_SEEDS,
    "a4_nn_min": common.SEEDS,
    "a5_nn_full": common.SEEDS,
    "blind": common.SEEDS,
}
"""Seeds per arm; every other arm uses the first three seeds."""
SHAPE_ARMS = ("a3_gam_full", "a5_nn_full")
GRID_POINTS = 25
MULTIPLIERS = np.linspace(0.5, 1.5, 13)
MONOTONE_TOLERANCE = 1e-4


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


def lag_residual(*, f: common.Farm, mu: np.ndarray) -> np.ndarray:
    """Return observed power one day earlier minus the plant forecast then (0 when unobserved)."""
    t = f.frame["time"].dt.epoch("us").to_numpy()
    days = np.where(f.frame["lead_hours"].to_numpy() <= 24, 1, 2)
    lag_t = t - days * 86_400_000_000
    idx = np.minimum(np.searchsorted(t, lag_t), len(t) - 1)
    found = (t[idx] == lag_t) & (f.frame["lag_missing"].to_numpy() == 0)
    return np.where(found, f.frame["lag_power"].to_numpy() - mu[idx], 0.0)


def permute_extras(*, f: common.Farm, mode: str, seed: int = 272) -> np.ndarray:
    """Return the full variable matrix with the extra columns shuffled or placebo-permuted."""
    x = f.frame.select(list(common.FULL_VARS)).to_numpy().astype(np.float64).copy()
    if mode == "real":
        return x
    rng = np.random.default_rng(seed)
    split = np.where(f.tuning, 0, np.where(f.scored, 1, 2))
    if mode == "shuffle":
        group = split
    else:
        month = f.frame["time"].dt.month().to_numpy()
        group = split * 1000 + month * 10 + np.digitize(f.frame["csi"].to_numpy(), CSI_BANDS)
    n_min = len(common.MIN_VARS)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        order = rng.permutation(idx)
        x[idx, n_min:] = x[order, n_min:]
    return x


def farm_arrays(
    *, f: common.Farm, scale: float, ceiling: float, mode: str
) -> dict[str, np.ndarray]:
    """Return a farm's variable matrix, plant inputs, head, and blind features at the setting's scale."""
    frame = f.frame
    mu = common.plant_forecast(farm=f, scale=scale, ceiling=ceiling)
    x = permute_extras(f=f, mode=mode)
    head = np.column_stack(
        [
            frame["relative_spread"].to_numpy(),
            frame["ghi_std"].to_numpy(),
            lag_residual(f=f, mu=mu),
            frame["lag_missing"].to_numpy(),
            frame["elevation_deg"].to_numpy(),
            frame["csi"].to_numpy(),
            frame["lead_hours"].to_numpy(),
            frame["temperature_mean"].to_numpy(),
        ]
    )
    blind = np.column_stack(
        [
            frame.select(list(common.RAW_FEATURES)).to_numpy().astype(np.float64),
            x[:, len(common.MIN_VARS) :],
        ]
    )
    return {
        "x": x,
        "mu": mu,
        "head": head,
        "blind": blind,
        "dni": frame["dni_mean"].to_numpy(),
        "dhi": frame["dhi_mean"].to_numpy(),
        "temperature": frame["temperature_mean"].to_numpy(),
        "zenith": frame["zenith_deg"].to_numpy(),
        "azimuth": frame["sun_azimuth_deg"].to_numpy(),
        "output": frame["output"].to_numpy().astype(np.float64),
    }


class Setting:
    """The train, validation, and test arrays of one (setting, scored farm, control mode)."""

    def __init__(self, *, setting: str, site: str, mode: str = "real") -> None:
        self.site, self.mode = site, mode
        others = [s for s in common.FARMS if s != site]
        train_sites = [site] if setting == "own_history" else others
        if setting == "own_history":
            self.scale, self.ceiling = farm(site).own_scale, farm(site).own_ceiling
        else:
            self.scale = float(np.median([farm(s).own_scale for s in others]))
            self.ceiling = float(np.median([farm(s).own_ceiling for s in others]))
        arrays = {
            s: farm_arrays(f=farm(s), scale=self.scale, ceiling=self.ceiling, mode=mode)
            for s in {*train_sites, site}
        }

        def gather(
            mask_fn: Callable[[common.Farm], np.ndarray], sites: list[str]
        ) -> dict[str, np.ndarray]:
            parts = [{k: v[mask_fn(farm(s))] for k, v in arrays[s].items()} for s in sites]
            return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

        self.train = gather(lambda f: f.tuning & ~f.validation, train_sites)
        self.val = gather(lambda f: f.tuning & f.validation, train_sites)
        self.test = gather(lambda f: f.scored, [site])
        self.test_frame = farm(site).frame.filter(pl.Series(farm(site).scored))
        mu_train = np.concatenate([self.train["mu"], self.val["mu"]])
        out_train = np.concatenate([self.train["output"], self.val["output"]])
        # Bin edges from all tuning rows; the initial table from the training rows only.
        self.edges, _ = common.bin_offsets(mu=mu_train, output=out_train)
        _, self.table = common.bin_offsets(mu=self.train["mu"], output=self.train["output"])
        self.mean = {k: self.train[k].mean(axis=0) for k in ("x", "head", "blind")}
        self.std = {k: self.train[k].std(axis=0) + 1e-6 for k in ("x", "head", "blind")}

    def knots(self, columns: list[int]) -> torch.Tensor:
        """Return 8 knots per variable at quantiles of the standardised training rows."""
        z = (self.train["x"][:, columns] - self.mean["x"][columns]) / self.std["x"][columns]
        k = np.quantile(z, np.linspace(0.02, 0.98, models.N_KNOTS), axis=0).T
        k = k + 1e-3 * np.arange(models.N_KNOTS)[None, :]
        return torch.tensor(k, dtype=torch.float32)

    def batch(self, *, split: str, columns: list[int]) -> dict[str, torch.Tensor]:
        """Return a split's tensors."""
        ds = getattr(self, split)
        n = len(ds["mu"])

        def t(a, dtype=torch.float32):  # noqa: ANN001, ANN202
            return torch.tensor(a, dtype=dtype)

        z = lambda key: (ds[key] - self.mean[key]) / self.std[key]  # noqa: E731
        return {
            "x": t(z("x")[:, columns]),
            "head": t(z("head")),
            "blind": t(z("blind")),
            "mu": t(ds["mu"]),
            "bin": torch.tensor(np.digitize(ds["mu"], self.edges), dtype=torch.long),
            "ceiling": torch.full((n,), self.ceiling),
            "scale": torch.full((n,), self.scale),
            "dni": t(ds["dni"]),
            "dhi": t(ds["dhi"]),
            "temperature": t(ds["temperature"]),
            "zenith_rad": t(np.radians(ds["zenith"])),
            "sun_azimuth_rad": t(np.radians(ds["azimuth"])),
            "actual": t(ds["output"]),
        }


def pinball(*, q: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over rows and the 11 levels."""
    levels = torch.tensor(common.LEVELS, dtype=torch.float32)
    error = actual[:, None] - q
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def run_epochs(
    *,
    model: torch.nn.Module,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    params: list[torch.nn.Parameter],
    epochs: int,
    lr: float,
    weight_decay: float,
    penalty: float,
    seed: int,
) -> float:
    """Train with pinball loss, keep the epoch with the lowest validation loss, return that loss."""
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = len(train["mu"])
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH):
            batch = {k: v[order[start : start + BATCH]] for k, v in train.items()}
            loss = pinball(q=model(batch), actual=batch["actual"])
            if penalty > 0.0:
                loss = loss + penalty * model.correction.penalty()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(pinball(q=model(val), actual=val["actual"]))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return best


def refit_table(*, model: models.PlantAdapter, st: Setting, val: dict[str, torch.Tensor]) -> None:
    """Replace the offset table with the per-bin residual quantiles of the frozen arm on all tuning rows.

    `val` holds the validation months; the training months are added here. Fitting the table on the
    two validation months alone gave a seasonal mismatch, so every arm uses all tuning rows.
    """
    train = st.batch(split="train", columns=list(range(val["x"].shape[1])))
    val = {k: torch.cat([train[k], val[k]]) for k in val}
    with torch.no_grad():
        power = model.power(val).numpy().astype(np.float64)
    residual = val["actual"].numpy().astype(np.float64) - power
    bins = val["bin"].numpy()
    overall = np.quantile(residual, common.LEVELS)
    table = np.empty((len(st.table), len(common.LEVELS)))
    for b in range(len(st.table)):
        rows = bins == b
        table[b] = (
            np.quantile(residual[rows], common.LEVELS)
            if rows.sum() >= common.MIN_BIN_ROWS
            else overall
        )
    with torch.no_grad():
        model.offsets.copy_(torch.tensor(table, dtype=torch.float32))


def fit_adapter(
    *, kind: str, st: Setting, columns: list[int], seed: int, penalty_index: int | None, alt: bool
) -> tuple[models.PlantAdapter, float, int, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Train an adapter over the penalty candidates, return it with its validation loss and penalty index."""
    train, val = st.batch(split="train", columns=columns), st.batch(split="val", columns=columns)
    candidates = range(2) if penalty_index is None else [penalty_index]
    best: tuple[float, int, models.PlantAdapter] | None = None
    for index in candidates:
        torch.manual_seed(seed)
        correction: torch.nn.Module = (
            models.AdditiveCurves(st.knots(columns))
            if kind == "gam"
            else models.NeuralCorrection(len(columns))
        )
        model = models.PlantAdapter(
            correction=correction, offsets=torch.tensor(st.table, dtype=torch.float32)
        )
        penalty, decay = PENALTIES[kind][index]
        params = [p for p in model.parameters() if p.requires_grad]
        score = run_epochs(
            model=model,
            train=train,
            val=val,
            params=params,
            epochs=EPOCHS,
            lr=LEARNING_RATE[kind],
            weight_decay=decay,
            penalty=penalty,
            seed=seed,
        )
        if best is None or score < best[0]:
            best = (score, index, model)
    assert best is not None
    score, index, model = best
    if alt:
        penalty, decay = PENALTIES[kind][index]
        for _ in range(2):
            model.log_scale.requires_grad_(True)
            run_epochs(
                model=model,
                train=train,
                val=val,
                params=[model.log_scale, model.offsets],
                epochs=ALT_EPOCHS,
                lr=LEARNING_RATE[kind],
                weight_decay=0.0,
                penalty=0.0,
                seed=seed,
            )
            model.log_scale.requires_grad_(False)
            params = [p for p in model.correction.parameters() if p.requires_grad]
            run_epochs(
                model=model,
                train=train,
                val=val,
                params=params,
                epochs=ALT_EPOCHS,
                lr=LEARNING_RATE[kind],
                weight_decay=decay,
                penalty=penalty,
                seed=seed,
            )
    refit_table(model=model, st=st, val=val)
    return model, score, index, train, val


def fit_head(
    *, adapter: models.PlantAdapter, st: Setting, columns: list[int], seed: int
) -> tuple[models.AdapterWithHead, float]:
    """Train experiment 243's H5-form head on a frozen adapter and return it with its validation loss."""
    train, val = st.batch(split="train", columns=columns), st.batch(split="val", columns=columns)
    torch.manual_seed(seed)
    model = models.AdapterWithHead(adapter=adapter, n_head_in=train["head"].shape[1] + 1)
    adapter.eval()
    params = list(model.net.parameters())
    best = float("inf")
    optimiser = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    n, best_state = len(train["mu"]), None
    for _ in range(EPOCHS):
        model.net.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH):
            batch = {k: v[order[start : start + BATCH]] for k, v in train.items()}
            loss = pinball(q=model(batch), actual=batch["actual"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(pinball(q=model(val), actual=val["actual"]))
        if score < best:
            best, best_state = (
                score,
                {k: v.detach().clone() for k, v in model.net.state_dict().items()},
            )
    assert best_state is not None
    model.net.load_state_dict(best_state)
    return model, best


def identity_adapter(*, st: Setting, columns: list[int], kind: str) -> models.PlantAdapter:
    """Return the plant at identity with the shared offset table fitted on validation rows."""
    correction: torch.nn.Module = (
        models.AdditiveCurves(st.knots(columns))
        if kind == "gam"
        else models.NeuralCorrection(len(columns))
    )
    model = models.PlantAdapter(
        correction=correction, offsets=torch.tensor(st.table, dtype=torch.float32)
    )
    refit_table(model=model, st=st, val=st.batch(split="val", columns=columns))
    return model


def predict(
    *, model: torch.nn.Module, st: Setting, columns: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scored farm's sorted (rows, 11) quantiles and the per-row crossing flag."""
    model.eval()
    with torch.no_grad():
        q, crossed = model.quantiles(st.batch(split="test", columns=columns))
    return q.numpy().astype(np.float64), crossed.numpy().astype(np.float64)


def shapes(*, model: models.PlantAdapter, st: Setting, arm: str, setting: str) -> pl.DataFrame:
    """Return g and h as functions of each variable (others averaged over the training rows)."""
    columns = list(range(len(common.FULL_VARS)))
    train = st.batch(split="train", columns=columns)
    rows = []
    for j, name in enumerate(common.FULL_VARS):
        raw = st.train["x"][:, j]
        grid_raw = np.linspace(*np.quantile(raw, [0.02, 0.98]), GRID_POINTS)
        for value in grid_raw:
            x = train["x"].clone()
            x[:, j] = float((value - st.mean["x"][j]) / st.std["x"][j])
            with torch.no_grad():
                g, h = model.gh({**train, "x": x})
            rows.append(
                {
                    "arm": arm,
                    "setting": setting,
                    "farm": st.site,
                    "variable": name,
                    "value": float(value),
                    "g_mean": float(g.mean()),
                    "h_mean": float(h.mean()),
                    "factor_mean": float(torch.exp(g).mean()),
                }
            )
    return pl.DataFrame(rows)


def monotone_share(*, model: models.PlantAdapter, st: Setting, columns: list[int]) -> float:
    """Return the share of grid steps where power falls as the ensemble irradiance multiplier rises."""
    base = st.batch(split="test", columns=columns)
    powers = []
    csi_index = common.FULL_VARS.index("csi")
    for m in MULTIPLIERS:
        x = base["x"].clone()
        raw_csi = st.test["x"][:, csi_index] * m
        x[:, columns.index(csi_index)] = torch.tensor(
            (raw_csi - st.mean["x"][csi_index]) / st.std["x"][csi_index], dtype=torch.float32
        )
        batch = {**base, "x": x, "dni": base["dni"] * float(m), "dhi": base["dhi"] * float(m)}
        with torch.no_grad():
            powers.append(model.power(batch).numpy())
    p = np.stack(powers)
    below_ceiling = p[1:] < st.ceiling - 1e-6
    falls = (p[1:] < p[:-1] - MONOTONE_TOLERANCE) & below_ceiling
    return float(falls.sum() / max(below_ceiling.sum(), 1))


def out_path(*, setting: str, site: str, arm: str, seed: int):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s{seed}_{site}.parquet"


def write(
    *,
    st: Setting,
    setting: str,
    arm: str,
    seed: int,
    quantiles: np.ndarray,
    crossed: np.ndarray,
    val_loss: float,
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
        val_loss=pl.lit(val_loss),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_arm(*, arm: str, setting: str, site: str, seed: int, cache_st: dict[str, Setting]) -> None:
    """Run one (arm, seed) job and write its files."""
    if out_path(setting=setting, site=site, arm=arm, seed=seed).exists():
        return
    mode = (
        "shuffle" if arm.endswith("shuffle") else "placebo" if arm.endswith("placebo") else "real"
    )
    if mode not in cache_st:
        cache_st[mode] = Setting(setting=setting, site=site, mode=mode)
    st = cache_st[mode]
    all_cols = list(range(len(common.FULL_VARS)))
    min_cols = list(range(len(common.MIN_VARS)))
    kind = "gam" if arm.startswith(("a2", "a3")) else "nn"
    columns = min_cols if arm in ("a2_gam_min", "a4_nn_min") else all_cols
    # Controls and heads reuse the penalty the base arm chose, found by retraining the base arm.
    base_name = {"a3_head": "a3_gam_full", "a5_head": "a5_nn_full"}.get(arm)
    penalty_index = None
    if arm in ("a3_shuffle", "a3_placebo", "a3_alt", "a5_shuffle", "a5_placebo", "a5_alt"):
        base = "a3_gam_full" if arm.startswith("a3") else "a5_nn_full"
        penalty_index = int(
            pl.read_parquet(common.DATA_DIR / f"penalty_{setting}_{site}_{base}_{seed}.parquet")[
                "index"
            ][0]
        )
    if arm == "blind":
        train, val = (
            st.batch(split="train", columns=all_cols),
            st.batch(split="val", columns=all_cols),
        )
        torch.manual_seed(seed)
        net = models.PureNetwork(train["blind"].shape[1])
        model = _BlindWrapper(net)
        score = run_epochs(
            model=model,
            train=train,
            val=val,
            params=list(net.parameters()),
            epochs=EPOCHS,
            lr=2e-3,
            weight_decay=1e-4,
            penalty=0.0,
            seed=seed,
        )
        q, crossed = predict(model=net, st=st, columns=all_cols)
    elif arm == "a1_head":
        adapter = identity_adapter(st=st, columns=all_cols, kind="nn")
        model, score = fit_head(adapter=adapter, st=st, columns=all_cols, seed=seed)
        q, crossed = predict(model=model, st=st, columns=all_cols)
    elif arm.endswith("_head"):
        assert base_name is not None
        adapter, _, index, _, _ = fit_adapter(
            kind=kind, st=st, columns=all_cols, seed=seed, penalty_index=None, alt=False
        )
        model, score = fit_head(adapter=adapter, st=st, columns=all_cols, seed=seed)
        q, crossed = predict(model=model, st=st, columns=all_cols)
    else:
        adapter, score, index, _, _ = fit_adapter(
            kind=kind,
            st=st,
            columns=columns,
            seed=seed,
            penalty_index=penalty_index,
            alt=arm.endswith("_alt"),
        )
        q, crossed = predict(model=adapter, st=st, columns=columns)
        if arm in ("a3_gam_full", "a5_nn_full"):
            row = pl.DataFrame(
                {"setting": setting, "farm": site, "arm": arm, "seed": seed, "index": index}
            )
            tmp = common.DATA_DIR / f"penalty_{setting}_{site}_{arm}_{seed}.parquet"
            row.write_parquet(tmp)
        if arm in SHAPE_ARMS and seed == 0:
            side = common.DATA_DIR / "sidecars"
            side.mkdir(parents=True, exist_ok=True)
            if setting == "own_history":
                shapes(model=adapter, st=st, arm=arm, setting=setting).write_parquet(
                    side / f"shapes_{arm}_{site}.parquet"
                )
            share = monotone_share(model=adapter, st=st, columns=columns)
            pl.DataFrame(
                {"arm": arm, "setting": setting, "farm": site, "violation_share": share}
            ).write_parquet(side / f"monotone_{arm}_{setting}_{site}.parquet")
    write(st=st, setting=setting, arm=arm, seed=seed, quantiles=q, crossed=crossed, val_loss=score)


class _BlindWrapper(torch.nn.Module):
    """Gives the blind network the `correction` attribute `run_epochs` expects."""

    def __init__(self, net: models.PureNetwork) -> None:
        super().__init__()
        self.net = net
        self.correction = net

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch)


def write_identity(*, setting: str, site: str) -> None:
    """Write A1': the plant at identity with the shared validation-fitted table."""
    if out_path(setting=setting, site=site, arm="a1_identity", seed=0).exists():
        return
    st = Setting(setting=setting, site=site)
    cols = list(range(len(common.FULL_VARS)))
    q, crossed = predict(
        model=identity_adapter(st=st, columns=cols, kind="nn"), st=st, columns=cols
    )
    write(
        st=st,
        setting=setting,
        arm="a1_identity",
        seed=0,
        quantiles=q,
        crossed=crossed,
        val_loss=float("nan"),
    )


def run_group(task: tuple[str, str, str]) -> str:
    """Run every arm and seed of a phase for one (setting, farm)."""
    torch.set_num_threads(1)
    phase, setting, site = task
    started = time.time()
    cache_st: dict[str, Setting] = {}
    arms = PHASE1 if phase == "phase1" else PHASE2
    if "E272_ARMS" in os.environ:
        arms = tuple(os.environ["E272_ARMS"].split(","))
    if phase == "phase1":
        write_identity(setting=setting, site=site)
    for arm in arms:
        for seed in ARM_SEEDS.get(arm, common.GAM_SEEDS):
            run_arm(arm=arm, setting=setting, site=site, seed=seed, cache_st=cache_st)
            print(f"{setting} {site} {arm} s{seed} {time.time() - started:.0f}s", flush=True)
    return f"{setting} {site}"


def main() -> None:
    """Run all (setting, farm) groups of the named phase in parallel."""
    phase = sys.argv[1]
    only = sys.argv[2:]
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(phase, s, f) for s, f in itertools.product(common.SETTINGS, common.FARMS)]
    if only:
        tasks = [t for t in tasks if t[2] in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, tasks):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
