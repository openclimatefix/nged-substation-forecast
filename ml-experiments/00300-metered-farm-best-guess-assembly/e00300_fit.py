# ruff: noqa: D102, D107, E501
"""Fit every arm of experiment 300 with monthly refits, in both settings.

For each scored month (2025-09 to 2026-08) every model is refitted on the rows whose valid time is
at least one day before the month starts, so no model sees an output that was unknown when the
month's first run was made. `own_history` trains on the scored farm only and `zero_history` on the
other five farms only. Arms: `p2` (spread-binned plant), `gam` (depth-1 boosted trees on P2's
quantiles as base margin), three GAM ablations (`gam_nospread`, `gam_nolag`, `gam_noavail`), `xgb`
(XGBoost, depth 6, same inputs as the GAM, no base margin), and `mdn` (three-component Gaussian
mixture, five seeds). One job is one (setting, farm, month); its files are written atomically and
skipped when they exist.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00300-*/e00300_fit.py
"""

import itertools
import multiprocessing
import sys
import time
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path

import e00300_common as common
import e00300_features as feats
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from scipy.special import ndtr
from studies.tail_calibration import level_column
from torch import nn  # ty: ignore[unresolved-import]

N_PROCESSES = 4
SEEDS = common.SEEDS
EPOCHS = 60
BATCH = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GAM_ROUNDS = 400
XGB_ROUNDS = 300
N_MU_BINS = 6
N_SPREAD_BINS = 3
COMPONENTS = 3
HEAD_INPUTS = ("mu", "plant_spread", "avail_ratio", "lag_power", "lag_missing", "csi", "lead_hours")
GAM_ARMS: dict[str, tuple[str, ...]] = {
    "gam": (),
    "gam_nospread": ("ghi_std", "plant_spread"),
    "gam_nolag": ("lag_power", "lag_missing"),
    "gam_noavail": ("avail_ratio",),
}
"""Each GAM arm and the columns it drops from experiment 270's rung-six inputs."""
SPREAD_FREE_ARMS: frozenset[str] = frozenset({"gam_nospread"})
"""Arms whose P2 base uses one spread bin."""


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


class Setting:
    """The training rows and scored rows of one (setting, farm, month), with P2 fitted."""

    def __init__(self, *, setting: str, site: str, start: datetime, n_spread_bins: int) -> None:
        self.site = site
        self.n_spread_bins = n_spread_bins
        cutoff = start - timedelta(days=common.REFIT_GAP_DAYS)
        validation_start = cutoff - timedelta(days=common.VALIDATION_DAYS)
        others = [s for s in common.FARMS if s != site]
        train_sites = [site] if setting == "own_history" else others
        params = {s: common.scale_and_ceiling(farm=farm(s), cutoff=cutoff) for s in train_sites}
        if setting == "own_history":
            self.scale, self.ceiling = params[site]
        else:
            self.scale = float(np.median([p[0] for p in params.values()]))
            self.ceiling = float(np.median([p[1] for p in params.values()]))
        pieces = []
        for s in train_sites:
            f = farm(s)
            table = feats.feature_frame(farm=f, scale=self.scale, ceiling=self.ceiling)
            time_col = table["time"]
            table = table.with_columns(is_val=time_col >= validation_start)
            pieces.append(table.filter(time_col < cutoff))
        self.train = pl.concat(pieces)
        test = farm(site)
        table = feats.feature_frame(farm=test, scale=self.scale, ceiling=self.ceiling)
        end = common.month_end(start=start)
        self.test = table.filter((pl.col("time") >= start) & (pl.col("time") < end))
        self.fit_p2()
        self.train_base = self.p2_quantiles(frame=self.train)
        self.test_base = self.p2_quantiles(frame=self.test)

    def fit_p2(self) -> None:
        """Fit P2's cells: 6 forecast bins by `n_spread_bins` spread bins, 11 residual quantiles each."""
        mu = self.train["mu"].to_numpy()
        spread = self.train["plant_spread"].to_numpy()
        y = self.train["output"].to_numpy().astype(float)
        four_edges, fallback = common.bin_offsets(mu=mu, output=y)
        usable = mu > common.MIN_FORECAST_POWER
        self.mu_edges = np.quantile(mu[usable], np.linspace(0, 1, N_MU_BINS + 1)[1:-1])
        self.spread_edges = np.quantile(
            spread[usable], np.linspace(0, 1, self.n_spread_bins + 1)[1:-1]
        )
        mu_bin, sp_bin = np.digitize(mu, self.mu_edges), np.digitize(spread, self.spread_edges)
        coarse = np.digitize(mu, four_edges)
        residual = y - mu
        self.cells = np.empty((N_MU_BINS, self.n_spread_bins, len(common.LEVELS)))
        for i in range(N_MU_BINS):
            for j in range(self.n_spread_bins):
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


def gam_quantiles(*, st: Setting, drop: tuple[str, ...], seed: int) -> np.ndarray:
    """Fit depth-1 boosted quantile trees on P2's quantiles as base margin."""
    columns = [c for c in feats.rung_columns(rung=6) if c not in drop]
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


def xgb_quantiles(*, st: Setting, seed: int) -> np.ndarray:
    """Fit XGBoost (depth 6, no base margin) on the GAM's inputs plus the plant's unit response."""
    columns = [*feats.rung_columns(rung=6), "unit_mean"]
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=common.LEVELS,
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


def train_mdn(*, st: Setting, seed: int) -> np.ndarray:
    """Fit one mixture head on the training rows and return the test rows' (rows, 11) quantiles."""
    torch.manual_seed(seed)
    train, test = st.train, st.test
    train_x = train.select(HEAD_INPUTS).to_numpy().astype(np.float64)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    is_val = train["is_val"].to_numpy()

    def batch(frame: pl.DataFrame, rows: np.ndarray | None) -> dict[str, torch.Tensor]:
        pick = np.ones(frame.height, dtype=bool) if rows is None else rows
        x = frame.select(HEAD_INPUTS).to_numpy().astype(np.float64)
        return {
            "x": torch.tensor(((x - mean) / std)[pick], dtype=torch.float32),
            "mu": torch.tensor(frame["mu"].to_numpy()[pick], dtype=torch.float32),
            "actual": torch.tensor(
                frame["output"].to_numpy().astype(np.float64)[pick], dtype=torch.float32
            ),
        }

    fit_b, val_b, test_b = batch(train, ~is_val), batch(train, is_val), batch(test, None)
    model = MixtureHead(len(HEAD_INPUTS))
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    n = len(fit_b["mu"])
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
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
    return mixture_quantiles(model=model, batch=test_b, ceiling=st.ceiling)


def out_path(*, setting: str, site: str, arm: str, seed: int, start: datetime) -> Path:
    """Return a job's output file."""
    month = start.strftime("%Y-%m")
    return common.PRED_DIR / setting / f"{arm}_s{seed}_{site}_{month}.parquet"


def write(
    *, st: Setting, setting: str, arm: str, seed: int, start: datetime, quantiles: np.ndarray
) -> None:
    """Write a job's test rows atomically."""
    path = out_path(setting=setting, site=st.site, arm=arm, seed=seed, start=start)
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


def run_job(task: tuple[str, str, datetime]) -> str:
    """Run every arm and seed for one (setting, farm, month), skipping outputs that exist."""
    torch.set_num_threads(1)
    setting, site, start = task
    started = time.time()
    standard = Setting(setting=setting, site=site, start=start, n_spread_bins=N_SPREAD_BINS)
    jobs: list[tuple[str, int]] = [("p2", 0), ("xgb", 0), *[(a, 0) for a in GAM_ARMS]]
    jobs += list(itertools.product(("mdn",), SEEDS))
    no_spread: Setting | None = None
    for arm, seed in jobs:
        if out_path(setting=setting, site=site, arm=arm, seed=seed, start=start).exists():
            continue
        st = standard
        if arm in SPREAD_FREE_ARMS:
            no_spread = no_spread or Setting(
                setting=setting, site=site, start=start, n_spread_bins=1
            )
            st = no_spread
        if arm == "p2":
            q = st.clamp(st.test_base)
        elif arm == "xgb":
            q = xgb_quantiles(st=st, seed=seed)
        elif arm in GAM_ARMS:
            q = gam_quantiles(st=st, drop=GAM_ARMS[arm], seed=seed)
        else:
            q = train_mdn(st=st, seed=seed)
        write(st=st, setting=setting, arm=arm, seed=seed, start=start, quantiles=q)
    return f"{setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def main() -> None:
    """Run all (setting, farm, month) jobs in parallel; arguments name settings, farms, or months."""
    tasks = list(itertools.product(common.SETTINGS, common.FARMS, common.REFIT_MONTHS))
    only = sys.argv[1:]
    if only:
        tasks = [t for t in tasks if t[0] in only or t[1] in only or f"{t[2]:%Y-%m}" in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_job, tasks):
            print("finished", done, flush=True)


if __name__ == "__main__":
    main()
