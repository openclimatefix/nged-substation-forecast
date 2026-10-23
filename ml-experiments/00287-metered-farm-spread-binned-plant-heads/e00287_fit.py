# ruff: noqa: D102, D107
"""Fit every arm of experiment 287 in both settings and write each job's scored-year quantiles.

Arms: `p2` (the plant with offsets binned by the forecast and the ensemble spread), `gam` (depth-1
boosted trees on experiment 270's rung-six inputs with P2's quantiles as base margin), `nn` (a
network whose output is P2's quantiles plus monotone corrections), `mdn` (a three-component Gaussian
mixture on the residual from the plant forecast), and three controls on `nn`. Settings as in
experiment 270: `own_history` trains on the scored farm's months before 2025-09 and `zero_history`
on the other five farms'. One job is one (setting, farm, arm, seed); its file is written atomically
and skipped when it exists.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00287-*/e00287_fit.py
"""

import itertools
import multiprocessing
import sys
import time
from functools import cache
from pathlib import Path

import e00287_common as common
import e00287_features as feats
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from scipy.special import ndtr
from studies.tail_calibration import level_column
from torch import nn  # ty: ignore[unresolved-import]

N_PROCESSES = 4
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 60
BATCH = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
SHUFFLE_SEED = 287
GAM_ROUNDS = 400
N_MU_BINS, N_SPREAD_BINS = 6, 3
MIN_GAP = 1e-3
COMPONENTS = 3
HEAD_INPUTS = ("mu", "plant_spread", "avail_ratio", "lag_power", "lag_missing", "csi", "lead_hours")
ARMS = {
    "nn": {"drop": (), "shuffle": ()},
    "mdn": {"drop": (), "shuffle": ()},
    "ctrl_spread": {"drop": (), "shuffle": ("plant_spread",)},
    "ctrl_lag": {"drop": (), "shuffle": ("lag_power", "lag_missing", "avail_ratio")},
    "ctrl_blind": {"drop": ("mu",), "shuffle": ()},
}
CONTROLS = ("ctrl_spread", "ctrl_lag", "ctrl_blind")


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


class Setting:
    """The training rows and scored rows of one (setting, scored farm), with P2 fitted."""

    def __init__(self, *, setting: str, site: str) -> None:
        self.site = site
        others = [s for s in common.FARMS if s != site]
        train_sites = [site] if setting == "own_history" else others
        if setting == "own_history":
            self.scale, self.ceiling = farm(site).own_scale, farm(site).own_ceiling
        else:
            self.scale = float(np.median([farm(s).own_scale for s in others]))
            self.ceiling = float(np.median([farm(s).own_ceiling for s in others]))
        pieces = []
        for s in train_sites:
            f = farm(s)
            table = feats.feature_frame(farm=f, scale=self.scale, ceiling=self.ceiling)
            table = table.with_columns(is_val=pl.Series(f.validation))
            pieces.append(table.filter(pl.Series(f.tuning)))
        self.train = pl.concat(pieces)
        test = farm(site)
        table = feats.feature_frame(farm=test, scale=self.scale, ceiling=self.ceiling)
        self.test = table.filter(pl.Series(test.scored))
        self.fit_p2()
        self.train_base = self.p2_quantiles(frame=self.train)
        self.test_base = self.p2_quantiles(frame=self.test)

    def fit_p2(self) -> None:
        """Fit P2's cells: 6 forecast bins by 3 spread bins, 11 residual quantiles each."""
        mu = self.train["mu"].to_numpy()
        spread = self.train["plant_spread"].to_numpy()
        y = self.train["output"].to_numpy().astype(float)
        four_edges, fallback = common.bin_offsets(mu=mu, output=y)
        usable = mu > common.MIN_FORECAST_POWER
        self.mu_edges = np.quantile(mu[usable], np.linspace(0, 1, N_MU_BINS + 1)[1:-1])
        self.spread_edges = np.quantile(spread[usable], np.linspace(0, 1, N_SPREAD_BINS + 1)[1:-1])
        mu_bin, sp_bin = np.digitize(mu, self.mu_edges), np.digitize(spread, self.spread_edges)
        coarse = np.digitize(mu, four_edges)
        residual = y - mu
        self.cells = np.empty((N_MU_BINS, N_SPREAD_BINS, len(common.LEVELS)))
        for i in range(N_MU_BINS):
            for j in range(N_SPREAD_BINS):
                rows = (mu_bin == i) & (sp_bin == j)
                self.cells[i, j] = (
                    np.quantile(residual[rows], common.LEVELS)
                    if rows.sum() >= common.MIN_BIN_ROWS
                    else fallback[int(np.bincount(coarse[rows]).argmax()) if rows.any() else 0]
                )

    def p2_quantiles(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return P2's sorted, unclamped quantiles for the frame's rows."""
        mu = frame["mu"].to_numpy()
        mu_bin = np.digitize(mu, self.mu_edges)
        sp_bin = np.digitize(frame["plant_spread"].to_numpy(), self.spread_edges)
        return np.sort(mu[:, None] + self.cells[mu_bin, sp_bin], axis=1)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Return the quantiles clamped to [0, ceiling]."""
        return np.minimum(np.maximum(q, 0.0), self.ceiling)


def gam_quantiles(*, st: Setting, seed: int) -> np.ndarray:
    """Fit depth-1 boosted quantile trees on P2's quantiles as base margin."""
    columns = feats.rung_columns(rung=6)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=common.LEVELS,
        tree_method="hist",
        n_estimators=GAM_ROUNDS,
        learning_rate=0.1,
        max_depth=1,
        min_child_weight=10,
        subsample=0.8,
        n_jobs=1,
        random_state=seed,
    )
    train_x = st.train.select(columns).to_numpy()
    model.fit(train_x, st.train["output"].to_numpy().astype(np.float64), base_margin=st.train_base)
    raw = np.asarray(
        model.predict(st.test.select(columns).to_numpy(), base_margin=st.test_base)
    ).reshape(st.test.height, -1)
    return st.clamp(np.sort(raw, axis=1))


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Return y with softplus(y) = x, for x > 0."""
    return x + torch.log(-torch.expm1(-x))


class MonotoneHead(nn.Module):
    """P2's lowest quantile plus a shift, then cumulative non-negative gaps; equals P2 at start."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 11)
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.net(batch["x"])
        base = batch["base"]
        first = base[:, :1] + 0.3 * torch.tanh(out[:, :1])
        gaps = torch.nn.functional.softplus(
            inverse_softplus(torch.diff(base, dim=1).clamp(min=MIN_GAP))
            + 2.0 * torch.tanh(out[:, 1:])
        )
        q = torch.cat([first, first + torch.cumsum(gaps, dim=1)], dim=1)
        return torch.minimum(torch.clamp(q, min=0.0), batch["ceiling"][:, None])


class MixtureHead(nn.Module):
    """Three-component Gaussian mixture on the residual from the plant forecast."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3 * COMPONENTS),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0] * 3 + [-0.1, 0.0, 0.1] + [float(np.log(0.08))] * 3))

    def parameters_of(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return (
            torch.log_softmax(out[:, :COMPONENTS], dim=1),
            out[:, COMPONENTS : 2 * COMPONENTS],
            torch.exp(out[:, 2 * COMPONENTS :].clamp(-6.0, 1.0)),
        )

    def nll(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        log_w, mean, std = self.parameters_of(batch["x"])
        z = (batch["actual"] - batch["mu"])[:, None]
        log_pdf = -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - 0.5 * float(np.log(2 * np.pi))
        return -torch.logsumexp(log_w + log_pdf, dim=1).mean()


def mixture_quantiles(
    *, model: MixtureHead, batch: dict[str, torch.Tensor], ceiling: float
) -> np.ndarray:
    """Return the mixture's 11 quantiles by bisection on its CDF, clamped to [0, ceiling]."""
    with torch.no_grad():
        log_w, mean, std = model.parameters_of(batch["x"])
    w, mean, std = (t.cpu().numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
    mu = batch["mu"].cpu().numpy().astype(np.float64)
    out = np.empty((len(mu), len(common.LEVELS)))
    for k, level in enumerate(common.LEVELS):
        lo, hi = np.full(len(mu), -2.0), np.full(len(mu), 2.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
            below = cdf < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        out[:, k] = 0.5 * (lo + hi)
    return np.minimum(np.maximum(mu[:, None] + out, 0.0), ceiling)


def shuffled(*, x: pl.DataFrame, columns: tuple[str, ...], seed: int) -> pl.DataFrame:
    """Return the frame with the named columns permuted together among rows."""
    order = np.random.default_rng(seed).permutation(x.height)
    return x.with_columns([pl.col(c).gather(pl.Series(order)) for c in columns])


def pinball(*, q: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over rows and the 11 levels."""
    error = actual[:, None] - q
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def train_neural(*, arm: str, st: Setting, seed: int) -> np.ndarray:
    """Fit one network on the tuning rows and return the scored rows' (rows, 11) quantiles."""
    torch.manual_seed(seed)
    spec = ARMS[arm]
    train, test = st.train, st.test
    if spec["shuffle"]:
        train = shuffled(x=train, columns=spec["shuffle"], seed=SHUFFLE_SEED + seed)
        test = shuffled(x=test, columns=spec["shuffle"], seed=SHUFFLE_SEED + 1000 + seed)
    inputs = [c for c in HEAD_INPUTS if c not in spec["drop"]]
    train_x = train.select(inputs).to_numpy().astype(np.float64)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    is_val = train["is_val"].to_numpy()

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32)

    def batch(
        frame: pl.DataFrame, base: np.ndarray, rows: np.ndarray | None
    ) -> dict[str, torch.Tensor]:
        pick = np.ones(frame.height, dtype=bool) if rows is None else rows
        x = frame.select(inputs).to_numpy().astype(np.float64)
        return {
            "x": tensor(((x - mean) / std)[pick]),
            "base": tensor(base[pick]),
            "mu": tensor(frame["mu"].to_numpy()[pick]),
            "actual": tensor(frame["output"].to_numpy().astype(np.float64)[pick]),
            "ceiling": torch.full((int(pick.sum()),), st.ceiling),
        }

    fit_b = batch(train, st.train_base, ~is_val)
    val_b = batch(train, st.train_base, is_val)
    test_b = batch(test, st.test_base, None)
    levels = tensor(common.LEVELS)
    mixture = arm == "mdn"
    model = MixtureHead(len(inputs)) if mixture else MonotoneHead(len(inputs))
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)

    def loss_of(b: dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(model, MixtureHead):
            return model.nll(b)
        return pinball(q=model(b), actual=b["actual"], levels=levels)

    best, best_state = float("inf"), None
    n = len(fit_b["mu"])
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            loss = loss_of({k: v[idx] for k, v in fit_b.items()})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(loss_of(val_b))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    if mixture:
        return mixture_quantiles(model=model, batch=test_b, ceiling=st.ceiling)  # ty: ignore[invalid-argument-type]
    with torch.no_grad():
        return np.sort(model(test_b).numpy().astype(np.float64), axis=1)


def out_path(*, setting: str, site: str, arm: str, seed: int) -> Path:
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s{seed}_{site}.parquet"


def write(*, st: Setting, setting: str, arm: str, seed: int, quantiles: np.ndarray) -> None:
    """Write a job's scored rows atomically."""
    path = out_path(setting=setting, site=st.site, arm=arm, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit(setting),
        actual=st.test["output"].cast(pl.Float64),
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
    jobs: list[tuple[str, int]] = [("p2", 0), ("gam", 0)]
    arms = ("nn", "mdn", *(CONTROLS if setting == "own_history" else ()))
    jobs += list(itertools.product(arms, SEEDS))
    for arm, seed in jobs:
        if out_path(setting=setting, site=site, arm=arm, seed=seed).exists():
            continue
        if arm == "p2":
            q = st.clamp(st.test_base)
        elif arm == "gam":
            q = gam_quantiles(st=st, seed=seed)
        else:
            q = train_neural(arm=arm, st=st, seed=seed)
        write(st=st, setting=setting, arm=arm, seed=seed, quantiles=q)
        print(f"{setting} {site} {arm} s{seed} done {time.time() - started:.0f}s", flush=True)
    return f"{setting} {site}"


def main() -> None:
    """Run all (setting, farm) groups in parallel; arguments name settings or farms to keep."""
    tasks = list(itertools.product(common.SETTINGS, common.FARMS))
    only = sys.argv[1:]
    if only:
        tasks = [t for t in tasks if t[0] in only or t[1] in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, tasks):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
