# ruff: noqa: E501, D107
"""Fit every arm of experiment 270 in both settings and write each job's scored-year quantiles.

Settings as in experiment 243: `own_history` fits on the scored farm's tuning months (before
2025-09); `zero_history` fits on the other five farms' tuning months, with the plant scale and
ceiling the median over those five farms. Every XGBoost arm has 11 quantile outputs, 150 rounds,
depth 5, learning rate 0.15, subsample 0.8, `colsample_bytree` 1 (equal across arms), and one fit per
seed. Quantiles are sorted per row before writing, and the raw crossing flag is stored. One job is
one (setting, farm, arm, seed); its output file is written atomically and skipped when it exists.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00270-*/e00270_fit.py
"""

import itertools
import multiprocessing
import sys
import time
from functools import cache

import e00270_common as common
import e00270_features as feats
import numpy as np
import polars as pl
import xgboost as xgb
from studies.tail_calibration import level_column

N_PROCESSES = 4
SEEDS = (0, 1, 2)
SHUFFLE_SEED = 270
GAM_ROUNDS = 400
"""Arms: name -> (columns, shuffled group or None, target). Rung `Lk` is `rung_columns(rung=k)`."""
ARMS: dict[str, tuple[list[str], tuple[str, ...], bool]] = {
    "x1_plant_inputs": (list(feats.PLANT_INPUTS), (), False),
    "x2_plant_forecast_only": (["mu"], (), False),
    **{f"l{k}": (feats.rung_columns(rung=k), (), False) for k in range(1, 7)},
    **{
        f"s{k}": (feats.rung_columns(rung=k), feats.GROUPS[feats.RUNGS[k - 1]], False)
        for k in range(2, 7)
    },
    "gam": (feats.rung_columns(rung=6), (), True),
}


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


class Setting:
    """The training and test arrays of one (setting, scored farm)."""

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
            pieces.append(table.filter(pl.Series(f.tuning)))
        self.train = pl.concat(pieces)
        test = farm(site)
        table = feats.feature_frame(farm=test, scale=self.scale, ceiling=self.ceiling)
        self.test = table.filter(pl.Series(test.scored))
        self.edges, self.table = common.bin_offsets(
            mu=self.train["mu"].to_numpy(), output=self.train["output"].to_numpy().astype(float)
        )


def plant_quantiles(*, st: Setting) -> np.ndarray:
    """Return the plant alone: forecast plus the tuning-month offsets, clamped to [0, ceiling]."""
    mu = st.test["mu"].to_numpy()
    q = mu[:, None] + common.offsets_for(mu=mu, edges=st.edges, table=st.table)
    return np.minimum(np.maximum(np.sort(q, axis=1), 0.0), st.ceiling)


def shuffled(*, x: pl.DataFrame, columns: tuple[str, ...], seed: int) -> pl.DataFrame:
    """Return the frame with the named columns permuted together among rows."""
    order = np.random.default_rng(seed).permutation(x.height)
    return x.with_columns([pl.col(c).gather(pl.Series(order)) for c in columns])


def xgboost_quantiles(
    *, st: Setting, columns: list[str], shuffle: tuple[str, ...], gam: bool, seed: int
) -> tuple[np.ndarray, float]:
    """Fit one XGBoost arm and return the sorted scored quantiles and the raw crossing rate."""
    train, test = st.train, st.test
    if shuffle:
        train = shuffled(x=train, columns=shuffle, seed=SHUFFLE_SEED)
        test = shuffled(x=test, columns=shuffle, seed=SHUFFLE_SEED + 1)
    train_x, test_x = train.select(columns).to_numpy(), test.select(columns).to_numpy()
    y = train["output"].to_numpy().astype(np.float64)
    mu_train, mu_test = train["mu"].to_numpy(), test["mu"].to_numpy()
    target = y - mu_train if gam else y
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=common.LEVELS,
        tree_method="hist",
        n_estimators=GAM_ROUNDS if gam else 150,
        learning_rate=0.1 if gam else 0.15,
        max_depth=1 if gam else 5,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=1.0,
        n_jobs=1,
        random_state=seed,
    )
    model.fit(train_x, target)
    raw = np.asarray(model.predict(test_x)).reshape(len(test_x), -1)
    if gam:
        raw = mu_test[:, None] + raw
    crossing = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
    out = np.maximum(np.sort(raw, axis=1), 0.0)
    return (np.minimum(out, st.ceiling) if gam else out), crossing


def out_path(*, setting: str, site: str, arm: str, seed: int):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s{seed}_{site}.parquet"


def write(
    *, st: Setting, setting: str, arm: str, seed: int, quantiles: np.ndarray, crossing: float
) -> None:
    """Write a job's scored rows atomically."""
    path = out_path(setting=setting, site=st.site, arm=arm, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit(setting),
        actual=st.test["output"].cast(pl.Float64),
        raw_crossing_rate=pl.lit(crossing),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_group(task: tuple[str, str]) -> str:
    """Run every arm and seed for one (setting, farm), skipping jobs whose file exists."""
    setting, site = task
    st = Setting(setting=setting, site=site)
    started = time.time()
    jobs = [("plant", 0)]
    for arm in ARMS:
        n_seeds = SEEDS if arm.startswith(("x", "l")) else (0,)
        jobs += [(arm, s) for s in n_seeds]
    for arm, seed in jobs:
        if out_path(setting=setting, site=site, arm=arm, seed=seed).exists():
            continue
        if arm == "plant":
            q, crossing = plant_quantiles(st=st), 0.0
        else:
            columns, shuffle, gam = ARMS[arm]
            q, crossing = xgboost_quantiles(
                st=st, columns=columns, shuffle=shuffle, gam=gam, seed=seed
            )
        write(st=st, setting=setting, arm=arm, seed=seed, quantiles=q, crossing=crossing)
        print(f"{setting} {site} {arm} s{seed} done {time.time() - started:.0f}s", flush=True)
    return f"{setting} {site}"


def main() -> None:
    """Run all (setting, farm) groups in parallel."""
    tasks = list(itertools.product(common.SETTINGS, common.FARMS))
    only = sys.argv[1:]
    if only:
        tasks = [t for t in tasks if t[1] in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, tasks):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
