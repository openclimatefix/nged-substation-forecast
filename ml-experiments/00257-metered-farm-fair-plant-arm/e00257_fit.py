# ruff: noqa: E501
"""Fit the convex quantile-regression arms of experiment 257 and write the scored-year quantiles.

P2 regresses each of the 11 offsets (observed output minus the plant forecast) on the plant
forecast, the 51 members' spread, a recent availability ratio, and lead, clear-sky and elevation
band dummies, one unpenalised linear programme per level
([Koenker and Bassett (1978)](https://doi.org/10.2307/1913643)). `own_history` fits on the scored
farm's own tuning months; `zero_history` fits on the other five farms' tuning months with the plant
scale and ceiling the median over those five farms. One job is one (setting, farm, arm); its
output file is written atomically and skipped when it exists. Run:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00257-*/e00257_fit.py
"""

import itertools
import multiprocessing
import sys
import time
from functools import cache

import e00257_common as common
import numpy as np
import polars as pl
from sklearn.linear_model import QuantileRegressor
from studies.tail_calibration import level_column

N_PROCESSES = 4
MIN_MU = 0.02
"""A lagged plant forecast below this share of capacity gives no availability ratio."""
RATIO_CLIP = 1.5
MEMBER_MEAN_TOLERANCE = 0.02
"""Largest mean absolute gap between the members' mean response and `unit_mean`."""
MEMBER_MEAN_MIN_CORRELATION = 0.99
"""Smallest correlation between the members' mean response and `unit_mean`."""
LEAD_EDGES = (18.0, 24.0, 30.0)
CSI_EDGES = (0.3, 0.6, 0.9)
ELEVATION_EDGES = (15.0, 30.0, 45.0)
ARMS = ("p2", "p2_no_ratio")


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


@cache
def members_unit(site: str) -> np.ndarray:
    """Return the 51 members' unit plant responses, rows aligned with `farm(site).frame`."""
    unit = np.load(common.E224_DIR / "members" / f"{site}_b0_unit.npy")
    raw = pl.read_parquet(common.E224_DIR / "members" / f"{site}_b0.parquet")
    if not raw["time"].is_sorted():
        msg = f"farm {site}: member rows are not sorted by time"
        raise ValueError(msg)
    # `unit_mean` is the plant response to the ensemble-mean weather, so it differs slightly from the
    # mean of the 51 members' responses, because the plant is non-linear.
    member_mean = unit.mean(axis=1)
    reference = raw["unit_mean"].to_numpy()
    if np.abs(member_mean - reference).mean() > MEMBER_MEAN_TOLERANCE:
        msg = f"farm {site}: member responses are not aligned with unit_mean"
        raise ValueError(msg)
    if np.corrcoef(member_mean, reference)[0, 1] < MEMBER_MEAN_MIN_CORRELATION:
        msg = f"farm {site}: member responses are not aligned with unit_mean"
        raise ValueError(msg)
    return unit


def availability(*, f: common.Farm, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the availability ratio and its missing flag.

    The ratio is the farm's power one day earlier (two days when the lead is over 24 hours) over the
    plant forecast at that earlier half-hour, clipped to 0 to `RATIO_CLIP`. It is 1 with the flag set
    when the earlier power is unobserved or the earlier plant forecast is under `MIN_MU`.
    """
    t = f.frame["time"].dt.epoch("us").to_numpy()
    days = np.where(f.frame["lead_hours"].to_numpy() <= 24, 1, 2)
    lag_t = t - days * 86_400_000_000
    idx = np.minimum(np.searchsorted(t, lag_t), len(t) - 1)
    found = (t[idx] == lag_t) & (f.frame["lag_missing"].to_numpy() == 0)
    mu_lag = mu[idx]
    usable = found & (mu_lag >= MIN_MU)
    ratio = np.where(
        usable,
        np.clip(f.frame["lag_power"].to_numpy() / np.maximum(mu_lag, MIN_MU), 0.0, RATIO_CLIP),
        1.0,
    )
    return ratio, (~usable).astype(np.float64)


def dummies(*, x: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Return one 0/1 column for every band except the first."""
    band = np.digitize(x, edges)
    return np.column_stack([(band == k).astype(np.float64) for k in range(1, len(edges) + 1)])


def design(
    *, f: common.Farm, scale: float, ceiling: float, with_ratio: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's design matrix and plant forecast at the setting's scale and ceiling."""
    frame = f.frame
    mu = common.plant_forecast(farm=f, scale=scale, ceiling=ceiling)
    spread = np.minimum(scale * members_unit(f.site).std(axis=1), ceiling)
    columns = [
        mu,
        spread,
        dummies(x=frame["lead_hours"].to_numpy(), edges=LEAD_EDGES),
        dummies(x=frame["csi"].to_numpy(), edges=CSI_EDGES),
        dummies(x=frame["elevation_deg"].to_numpy(), edges=ELEVATION_EDGES),
    ]
    if with_ratio:
        ratio, missing = availability(f=f, mu=mu)
        columns += [ratio, missing]
    x = np.column_stack(columns)
    return x, mu


def fit_and_predict(
    *, setting: str, site: str, with_ratio: bool
) -> tuple[pl.DataFrame, np.ndarray]:
    """Fit the 11 levels on the setting's tuning rows and return the scored farm's rows and quantiles."""
    others = [s for s in common.FARMS if s != site]
    train_sites = [site] if setting == "own_history" else others
    if setting == "own_history":
        scale, ceiling = farm(site).own_scale, farm(site).own_ceiling
    else:
        scale = float(np.median([farm(s).own_scale for s in others]))
        ceiling = float(np.median([farm(s).own_ceiling for s in others]))
    xs, ys = [], []
    for s in train_sites:
        f = farm(s)
        x, mu = design(f=f, scale=scale, ceiling=ceiling, with_ratio=with_ratio)
        output = f.frame["output"].to_numpy().astype(np.float64)
        xs.append(x[f.tuning])
        ys.append((output - mu)[f.tuning])
    train_x, train_y = np.concatenate(xs), np.concatenate(ys)
    centre, spread = train_x.mean(axis=0), train_x.std(axis=0) + 1e-9
    test = farm(site)
    test_x, test_mu = design(f=test, scale=scale, ceiling=ceiling, with_ratio=with_ratio)
    mask = test.scored
    offsets = np.empty((int(mask.sum()), len(common.LEVELS)))
    for k, level in enumerate(common.LEVELS):
        model = QuantileRegressor(quantile=float(level), alpha=0.0, solver="highs")
        model.fit((train_x - centre) / spread, train_y)
        offsets[:, k] = model.predict((test_x[mask] - centre) / spread)
    quantiles = np.minimum(
        np.maximum(np.sort(test_mu[mask][:, None] + offsets, axis=1), 0.0), ceiling
    )
    return test.frame.filter(pl.Series(mask)), quantiles


def out_path(*, setting: str, site: str, arm: str):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s0_{site}.parquet"


def write(*, frame: pl.DataFrame, setting: str, site: str, arm: str, quantiles: np.ndarray) -> None:
    """Write a job's scored rows atomically."""
    path = out_path(setting=setting, site=site, arm=arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(site),
        arm=pl.lit(arm),
        seed=pl.lit(0),
        setting=pl.lit(setting),
        actual=frame["output"].cast(pl.Float64),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)


def run_job(task: tuple[str, str, str]) -> str:
    """Run one (setting, farm, arm), skipping it when its file exists."""
    setting, site, arm = task
    if out_path(setting=setting, site=site, arm=arm).exists():
        return f"{setting} {site} {arm} (skipped)"
    started = time.time()
    frame, quantiles = fit_and_predict(setting=setting, site=site, with_ratio=arm == "p2")
    write(frame=frame, setting=setting, site=site, arm=arm, quantiles=quantiles)
    return f"{setting} {site} {arm} done {time.time() - started:.0f}s"


def main() -> None:
    """Run all jobs in parallel: both arms with own history, and P2 with zero history."""
    tasks = [
        (setting, site, arm)
        for setting, site, arm in itertools.product(common.SETTINGS, common.FARMS, ARMS)
        if setting == "own_history" or arm == "p2"
    ]
    only = sys.argv[1:]
    if only:
        tasks = [t for t in tasks if t[1] in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_job, tasks):
            print(done, flush=True)


if __name__ == "__main__":
    main()
