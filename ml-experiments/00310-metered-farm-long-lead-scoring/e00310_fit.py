# ruff: noqa: D102, D107, E501
"""Fit every arm of experiment 310 for each day band, setting, farm, and refit period.

Copied from experiment 300's fitter. A refit at `fit_start` trains on rows whose valid time is at
least one day before `fit_start`, and predicts the rows whose run (`init_time`) falls in
`[fit_start, fit_start + REFIT_MONTHS)`, so no model sees an output that was unknown when the run
was made. `own_history` trains on the scored farm only and `zero_history` on the other five farms
only. Arms: `p2` (spread-binned plant), `gam` (depth-1 boosted trees on P2's quantiles as base
margin), `xgb` (XGBoost, depth 6, no base margin), `mdn` (three-component Gaussian mixture), and the
`_nolag` copy of each model without the lagged-power columns (own history only, to save compute). One job is one (band, setting, farm,
refit); its files are written atomically and skipped when they exist.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00310-*/e00310_fit.py [band|setting|farm|YYYY-MM ...]
"""

import itertools
import multiprocessing
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00310_common as common
import e00310_features as feats
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from scipy.special import ndtr
from studies.tail_calibration import level_column
from torch import nn  # ty: ignore[unresolved-import]

N_PROCESSES: Final[int] = 4
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
EPOCHS = 40
BATCH = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GAM_ROUNDS = 400
XGB_ROUNDS = 300
N_MU_BINS = 6
N_SPREAD_BINS = 3
COMPONENTS = 3
REFIT_MONTHS: Final[int] = 3
"""Months between refits (screening schedule); each refit predicts the runs of its months."""
REFIT_STARTS: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (8 + i) // 12, (8 + i) % 12 + 1, 1, tzinfo=UTC)
    for i in range(0, 12, REFIT_MONTHS)
)
REFIT_GAP_DAYS: Final[int] = 1
VALIDATION_DAYS: Final[int] = 61
HEAD_INPUTS: Final[tuple[str, ...]] = (
    "mu",
    "plant_spread",
    "avail_ratio",
    "lag_power",
    "lag_missing",
    "csi",
    "lead_hours",
)
ARMS: Final[tuple[str, ...]] = (
    "p2",
    "xgb",
    "gam",
    "mdn",
    "xgb_nolag",
    "gam_nolag",
    "mdn_nolag",
)


def level_columns() -> list[str]:
    """Return the 11 quantile column names."""
    return [level_column(level=lv) for lv in common.LEVELS]


def month_add(*, start: datetime, months: int) -> datetime:
    """Return the first day of the month `months` after `start`."""
    index = start.year * 12 + start.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def columns_for(*, arm: str) -> list[str]:
    """Return an arm's feature columns."""
    drop = feats.LAG_COLUMNS if arm.endswith("_nolag") else ()
    return [c for c in feats.RUNG6 if c not in drop]


class Setting:
    """The training rows and scored rows of one (band, setting, farm, refit), with P2 fitted."""

    def __init__(
        self, *, band: str, setting: str, site: str, start: datetime, n_spread_bins: int
    ) -> None:
        self.site = site
        self.n_spread_bins = n_spread_bins
        cutoff = start - timedelta(days=REFIT_GAP_DAYS)
        validation_start = cutoff - timedelta(days=VALIDATION_DAYS)
        others = [s for s in common.FARMS if s != site]
        train_sites = [site] if setting == "own_history" else others
        farms = {s: feats.load_band_farm(s, band) for s in train_sites}
        params = {s: feats.scale_and_ceiling(farm=farms[s], cutoff=cutoff) for s in train_sites}
        if setting == "own_history":
            self.scale, self.ceiling = params[site]
        else:
            self.scale = float(np.median([p[0] for p in params.values()]))
            self.ceiling = float(np.median([p[1] for p in params.values()]))
        pieces = []
        for s in train_sites:
            table = feats.feature_frame(farm=farms[s], scale=self.scale, ceiling=self.ceiling)
            table = table.with_columns(is_val=pl.col("time") >= validation_start)
            pieces.append(table.filter(pl.col("time") < cutoff))
        self.train = pl.concat(pieces)
        test_farm = feats.load_band_farm(site, band)
        table = feats.feature_frame(farm=test_farm, scale=self.scale, ceiling=self.ceiling)
        end = month_add(start=start, months=REFIT_MONTHS)
        self.test = table.filter(
            (pl.col("init_time") >= start)
            & (pl.col("init_time") < end)
            & (pl.col("time") < common.SCORED_END)
        )
        self.fit_p2()
        self.train_base = self.p2_quantiles(frame=self.train)
        self.test_base = self.p2_quantiles(frame=self.test)

    def fit_p2(self) -> None:
        """Fit P2's cells: 6 forecast bins by `n_spread_bins` spread bins, 11 residual quantiles each."""
        mu = self.train["mu"].to_numpy()
        spread = self.train["plant_spread"].to_numpy()
        y = self.train["output"].to_numpy().astype(float)
        usable = mu > common.MIN_FORECAST_POWER
        four_edges = np.quantile(mu[usable], np.linspace(0, 1, common.N_BINS + 1)[1:-1])
        coarse = np.digitize(mu, four_edges)
        residual = y - mu
        overall = np.quantile(residual, common.LEVELS)
        fallback = np.empty((common.N_BINS, len(common.LEVELS)))
        for index in range(common.N_BINS):
            rows = coarse == index
            fallback[index] = (
                np.quantile(residual[rows], common.LEVELS)
                if rows.sum() >= common.MIN_BIN_ROWS
                else overall
            )
        self.mu_edges = np.quantile(mu[usable], np.linspace(0, 1, N_MU_BINS + 1)[1:-1])
        self.spread_edges = np.quantile(
            spread[usable], np.linspace(0, 1, self.n_spread_bins + 1)[1:-1]
        )
        mu_bin, sp_bin = np.digitize(mu, self.mu_edges), np.digitize(spread, self.spread_edges)
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


def gam_quantiles(*, st: Setting, arm: str, seed: int) -> np.ndarray:
    """Fit depth-1 boosted quantile trees on P2's quantiles as base margin."""
    columns = columns_for(arm=arm)
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


def xgb_quantiles(*, st: Setting, arm: str, seed: int) -> np.ndarray:
    """Fit XGBoost (depth 6, no base margin) on the GAM's inputs plus the plant's unit response."""
    columns = [*columns_for(arm=arm), "unit_mean"]
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


def train_mdn(*, st: Setting, arm: str, seed: int) -> np.ndarray:
    """Fit one mixture head on the training rows and return the test rows' (rows, 11) quantiles."""
    torch.manual_seed(seed)
    inputs = [c for c in HEAD_INPUTS if not (arm.endswith("_nolag") and c in feats.LAG_COLUMNS)]
    train, test = st.train, st.test
    train_x = train.select(inputs).to_numpy().astype(np.float64)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    is_val = train["is_val"].to_numpy()

    def batch(frame: pl.DataFrame, rows: np.ndarray | None) -> dict[str, torch.Tensor]:
        pick = np.ones(frame.height, dtype=bool) if rows is None else rows
        x = frame.select(inputs).to_numpy().astype(np.float64)
        return {
            "x": torch.tensor(((x - mean) / std)[pick], dtype=torch.float32),
            "mu": torch.tensor(frame["mu"].to_numpy()[pick], dtype=torch.float32),
            "actual": torch.tensor(
                frame["output"].to_numpy().astype(np.float64)[pick], dtype=torch.float32
            ),
        }

    fit_b, val_b, test_b = batch(train, ~is_val), batch(train, is_val), batch(test, None)
    model = MixtureHead(len(inputs))
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


def out_path(*, band: str, setting: str, site: str, arm: str, seed: int, start: datetime) -> Path:
    """Return a job's output file."""
    month = start.strftime("%Y-%m")
    return common.PRED_DIR / band / setting / f"{arm}_s{seed}_{site}_{month}.parquet"


def write(
    *,
    st: Setting,
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
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_job(task: tuple[str, str, str, datetime]) -> str:
    """Run every arm and seed for one (band, setting, farm, refit), skipping outputs that exist."""
    torch.set_num_threads(1)
    band, setting, site, start = task
    started = time.time()
    arms = [a for a in ARMS if setting == "own_history" or not a.endswith("_nolag")]
    jobs: list[tuple[str, int]] = [(a, 0) for a in arms if not a.startswith("mdn")]
    jobs += [(a, s) for a in arms if a.startswith("mdn") for s in SEEDS]
    todo = [
        j
        for j in jobs
        if not out_path(
            band=band, setting=setting, site=site, arm=j[0], seed=j[1], start=start
        ).exists()
    ]
    if not todo:
        return f"{band} {setting} {site} {start:%Y-%m} skipped"
    st = Setting(band=band, setting=setting, site=site, start=start, n_spread_bins=N_SPREAD_BINS)
    for arm, seed in todo:
        if arm == "p2":
            q = st.clamp(st.test_base)
        elif arm.startswith("xgb"):
            q = xgb_quantiles(st=st, arm=arm, seed=seed)
        elif arm.startswith("gam"):
            q = gam_quantiles(st=st, arm=arm, seed=seed)
        else:
            q = train_mdn(st=st, arm=arm, seed=seed)
        write(st=st, band=band, setting=setting, arm=arm, seed=seed, start=start, quantiles=q)
    return f"{band} {setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def main() -> None:
    """Run all jobs in parallel; arguments name bands, settings, farms, or refit months."""
    tasks = list(
        itertools.product(reversed(common.BAND_ORDER), common.SETTINGS, common.FARMS, REFIT_STARTS)
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
