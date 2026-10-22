# ruff: noqa: E501
"""Build every arm's 11 quantiles for each farm, fold, and lead band, and write them.

Arms (all in power over capacity):

| Arm | Centre(s) | Spread |
|---|---|---|
| `floor_heuristic` | 13 earlier observed powers (6 weekly, 7 annual analogues) | their empirical quantiles |
| `floor_climatology` | the farm's earlier powers in the same month, half-hour, and weekend cell | pooled quantiles |
| `a2_mean` | plant response to the ensemble mean | tuning residual quantiles in 4 bins of the forecast |
| `a3_two_number` | mean `mu` of the 51 responses | 51 Gaussian points `mu + s * sigma * z` times the offsets (`f`) |
| `a4_members` | the 51 responses | the offsets times `f` |
| `a2b_rescale` | `mu` only (placebo: every response replaced by the mean) | offsets times `f` |
| `p3_shuffle`, `p3_strong`, `p4_shuffle`, `p4_strong` | arms 3 and 4 with the spread taken from a random donor row | |
| `a5_xgboost` | XGBoost, 11 quantile outputs on ensemble-mean weather, spread, sun geometry, and lead | |

The offset factor `f` and the spread factor `s` are shared across farms and fitted
leave-one-farm-out by the daily-loss grid search of experiments 187 and 193 (`f` first with `s = 1`,
then `s`, then `f` again), on the other farms' tuning rows of the same lead band.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00224-*/e00224_arms.py
"""

import itertools
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import Final

import e00224_common as common
import e00224_members as members
import numpy as np
import polars as pl
import xgboost as xgb
from baseline_forecasters.climatology import CLIMATOLOGY_POOLING_OFFSETS
from scipy.stats import norm
from studies.ens_members import mixture_quantiles
from studies.tail_calibration import level_column

N_PROCESSES: Final[int] = 4
LEVELS: Final[np.ndarray] = np.array(common.LEVELS)
Z_POINTS: Final[np.ndarray] = norm.ppf((np.arange(common.N_MEMBERS) + 0.5) / common.N_MEMBERS)
F_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
S_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.5001, 0.1), 2)
FIT_ROWS: Final[int] = 10_000
CHUNK: Final[int] = 4000
N_BINS: Final[int] = 4
MIN_BIN_ROWS: Final[int] = 50
MIN_TUNING_ROWS: Final[int] = 500
MIN_FORECAST_POWER: Final[float] = 0.02
SEED: Final[int] = 224
HEURISTIC_LAG_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 49, 50, 51, 52, 53, 54, 55)
XGB_FEATURES: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "temperature_mean",
    "dni_mean",
    "dhi_mean",
    "unit_mean",
    "elevation_deg",
    "sun_azimuth_deg",
    "lead_hours",
)
ARM_NAMES: Final[tuple[str, ...]] = (
    "floor_heuristic",
    "floor_climatology",
    "a2_mean",
    "a2b_rescale",
    "a3_two_number",
    "a4_members",
    "a5_xgboost",
    "p3_shuffle",
    "p3_strong",
    "p4_shuffle",
    "p4_strong",
)


@dataclass
class FarmBand:
    """One farm's rows for one lead band and fold, with its plant response and offsets."""

    site: str
    band: int
    table: pl.DataFrame
    tuning: np.ndarray
    scored: np.ndarray
    output: np.ndarray
    mu0: np.ndarray
    response: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    bin0: np.ndarray
    bin_m: np.ndarray
    offsets0: np.ndarray
    offsets_m: np.ndarray


def _bin_offsets(
    *, forecast: np.ndarray, residual: np.ndarray, tuning: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's bin and the (rows, 11) residual quantiles of its bin."""
    usable = tuning & (forecast > MIN_FORECAST_POWER)
    edges = np.quantile(forecast[usable], np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.digitize(forecast, edges)
    overall = np.quantile(residual[tuning], LEVELS)
    table = np.empty((N_BINS, len(LEVELS)))
    for index in range(N_BINS):
        rows = tuning & (bins == index)
        table[index] = (
            np.quantile(residual[rows], LEVELS) if rows.sum() >= MIN_BIN_ROWS else overall
        )
    return bins, table[bins]


def build_farm_band(*, site: str, band: int, fold: str) -> FarmBand | None:
    """Fit the plant scale and ceiling on the tuning rows and return the farm-band arrays."""
    start, end = common.FOLDS[fold]
    table = pl.read_parquet(members.table_path(site=site, band=band))
    unit = np.load(members.unit_path(site=site, band=band)).astype(np.float64)
    time = table["time"]
    tuning = (time < start).to_numpy()
    scored = ((time >= start) & (time < end)).to_numpy()
    if tuning.sum() < MIN_TUNING_ROWS or scored.sum() == 0:
        return None
    output = table["output"].to_numpy().astype(np.float64)
    x = table["unit_mean"].to_numpy()
    scale = float(np.sum(output[tuning] * x[tuning]) / np.sum(x[tuning] ** 2))
    ceiling = float(np.quantile(output[tuning], common.CEILING_QUANTILE))
    mu0 = np.minimum(scale * x, ceiling)
    response = np.minimum(scale * unit, ceiling)
    mu = response.mean(axis=1)
    sigma = response.std(axis=1)
    bin0, offsets0 = _bin_offsets(forecast=mu0, residual=output - mu0, tuning=tuning)
    bin_m, offsets_m = _bin_offsets(forecast=mu, residual=output - mu, tuning=tuning)
    return FarmBand(
        site=site,
        band=band,
        table=table,
        tuning=tuning,
        scored=scored,
        output=output,
        mu0=mu0,
        response=response,
        mu=mu,
        sigma=sigma,
        bin0=bin0,
        bin_m=bin_m,
        offsets0=offsets0,
        offsets_m=offsets_m,
    )


def mixture(*, centres: np.ndarray, offsets: np.ndarray, f: float) -> np.ndarray:
    """Return 11 mixture quantiles of (rows, 51) centres, each with `f` times its row's offsets."""
    out = np.empty((len(centres), len(LEVELS)))
    for start in range(0, len(centres), CHUNK):
        part = slice(start, start + CHUNK)
        out[part] = mixture_quantiles(
            member_quantiles=centres[part, :, None] + f * offsets[part, None, :]
        )
    return out


def pinball(*, actual: np.ndarray, quantiles: np.ndarray) -> float:
    """Return the pinball loss averaged over the 11 levels and the rows."""
    error = actual[:, None] - quantiles
    return float(np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error)))


def centres_two_number(*, mu: np.ndarray, sigma: np.ndarray, s: float) -> np.ndarray:
    """Return 51 Gaussian points per row: `mu + s * sigma * z`."""
    return mu[:, None] + s * sigma[:, None] * Z_POINTS[None, :]


def fit_factors(
    *, pool: dict[str, np.ndarray], make_centres: Callable[[float], np.ndarray], fit_spread: bool
) -> tuple[float, float]:
    """Return `(f, s)` minimising the pinball loss on the pooled tuning rows."""

    def loss(f: float, s: float) -> float:
        quantiles = mixture(centres=make_centres(s), offsets=pool["offsets"], f=f)
        return pinball(actual=pool["actual"], quantiles=np.maximum(quantiles, 0.0))

    f, s = 1.0, 1.0
    f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
    if fit_spread:
        s = float(S_GRID[int(np.argmin([loss(f, g) for g in S_GRID]))])
        f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
    return f, s


def make_pool(*, farms: list[FarmBand], seed: int) -> dict[str, np.ndarray]:
    """Return up to `FIT_ROWS` tuning daylight rows pooled over the given farms."""
    rng = np.random.default_rng(seed)
    pieces = []
    for farm in farms:
        rows = np.flatnonzero(farm.tuning & (farm.mu > MIN_FORECAST_POWER))
        pieces.append((farm, rows))
    total = sum(len(rows) for _, rows in pieces)
    keep = min(FIT_ROWS, total)
    chosen = []
    for farm, rows in pieces:
        take = max(1, round(keep * len(rows) / total))
        chosen.append((farm, np.sort(rng.choice(rows, size=min(take, len(rows)), replace=False))))
    return {
        "actual": np.concatenate([f.output[r] for f, r in chosen]),
        "mu": np.concatenate([f.mu[r] for f, r in chosen]),
        "sigma": np.concatenate([f.sigma[r] for f, r in chosen]),
        "response": np.concatenate([f.response[r] for f, r in chosen]),
        "offsets": np.concatenate([f.offsets_m[r] for f, r in chosen]),
    }


def donors(*, farm: FarmBand, strong: bool, seed: int) -> np.ndarray:
    """Return a donor row per row: a random row of the same month and hour band (and forecast bin)."""
    table = farm.table
    key = (
        table["time"].dt.month().cast(pl.String)
        + "|"
        + (table["time"].dt.hour() // 3).cast(pl.String)
    )
    if strong:
        key = key + "|" + pl.Series(farm.bin_m.astype(str))
    key = key.to_numpy()
    source = np.arange(table.height)
    rng = np.random.default_rng(seed)
    scored = np.flatnonzero(farm.scored)
    for label in np.unique(key[scored]):
        rows = scored[key[scored] == label]
        source[rows] = rows[rng.permutation(len(rows))]
    return source


def frame_of(*, farm: FarmBand, fold: str, quantiles: np.ndarray, arm: str) -> pl.DataFrame:
    """Return the scored rows of one arm with the columns the scorer reads."""
    rows = farm.scored
    table = farm.table.filter(pl.Series(rows))
    clipped = np.maximum(np.sort(quantiles[rows].astype(np.float64), axis=1), 0.0)
    return table.select(
        time_series_id="site",
        power_fcst_init_time="init_time",
        valid_time="time",
        lead_hours="lead_hours",
        band="band",
        actual="output",
    ).with_columns(
        seed=pl.lit(0),
        actual=pl.col("actual").cast(pl.Float64),
        lead_hours=pl.col("lead_hours").cast(pl.Float64),
        fold=pl.lit(fold),
        arm=pl.lit(arm),
        **{level_column(level=lv): pl.Series(clipped[:, i]) for i, lv in enumerate(common.LEVELS)},
    )


def heuristic_and_climatology(
    *, farm: FarmBand, fold: str, full: pl.DataFrame
) -> dict[str, np.ndarray]:
    """Return the two floors' 11 quantiles for every row (NaN where the floor has no member)."""
    start, _ = common.FOLDS[fold]
    times = farm.table.select("time")
    history = full.filter(pl.col("site") == farm.site)
    columns = []
    for weeks in HEURISTIC_LAG_WEEKS:
        lagged = history.select(time=pl.col("time") + timedelta(weeks=weeks), lag=pl.col("output"))
        columns.append(times.join(lagged, on="time", how="left")["lag"].to_numpy())
    stack = np.column_stack(columns).astype(np.float64)
    count = np.sum(~np.isnan(stack), axis=1)
    heuristic = np.full((len(times), len(LEVELS)), np.nan)
    valid = count > 0
    heuristic[valid] = np.nanquantile(stack[valid], LEVELS, axis=1).T
    samples = (
        history.filter(pl.col("time") < start)
        .select("time", "output")
        .with_columns(local=pl.col("time").dt.convert_time_zone("Europe/London"))
    )
    keyed = (
        samples.select(
            "output",
            month=pl.col("local").dt.month(),
            half_hour=pl.col("local").dt.hour() * 2 + pl.col("local").dt.minute() // 30,
            weekend=pl.col("local").dt.weekday() >= 6,
        )
        .join(CLIMATOLOGY_POOLING_OFFSETS, how="cross")
        .select(
            "output",
            "weekend",
            month=((pl.col("month") - 1 + pl.col("month_offset")) % 12 + 1),
            half_hour=((pl.col("half_hour") + pl.col("half_hour_offset")) % 48),
        )
    )
    lookup = keyed.group_by("month", "half_hour", "weekend").agg(
        *[pl.col("output").quantile(lv, "linear").alias(f"c{i}") for i, lv in enumerate(LEVELS)]
    )
    local = farm.table.select(local=pl.col("time").dt.convert_time_zone("Europe/London")).select(
        month=pl.col("local").dt.month(),
        half_hour=pl.col("local").dt.hour() * 2 + pl.col("local").dt.minute() // 30,
        weekend=pl.col("local").dt.weekday() >= 6,
    )
    climatology = (
        local.join(lookup, on=["month", "half_hour", "weekend"], how="left")
        .select([f"c{i}" for i in range(len(LEVELS))])
        .to_numpy()
    )
    return {"floor_heuristic": heuristic, "floor_climatology": climatology.astype(np.float64)}


@cache
def farms_of(fold: str, band: int) -> dict[str, FarmBand]:
    """Build (once per process) every farm's arrays for a fold and band."""
    return {
        site: built
        for site in common.FARMS
        if (built := build_farm_band(site=site, band=band, fold=fold)) is not None
    }


@cache
def full_output() -> pl.DataFrame:
    """Return the metered output of every farm (cached per process)."""
    return common.farm_output()


def task(job: tuple[str, int, str]) -> list[pl.DataFrame]:
    """Fit the shared factors leave-one-farm-out for one farm-band and build its arm frames."""
    fold, band, site = job
    farms = farms_of(fold, band)
    farm = farms[site]
    others = [f for name, f in farms.items() if name != site]
    pool = make_pool(farms=others, seed=SEED + band)
    f_rescale, _ = fit_factors(
        pool=pool,
        make_centres=lambda s: np.repeat(pool["mu"][:, None], common.N_MEMBERS, axis=1),
        fit_spread=False,
    )
    f_members, _ = fit_factors(pool=pool, make_centres=lambda s: pool["response"], fit_spread=False)
    f_two, s_two = fit_factors(
        pool=pool,
        make_centres=lambda s: centres_two_number(mu=pool["mu"], sigma=pool["sigma"], s=s),
        fit_spread=True,
    )
    print("factors", fold, band, site, f_rescale, f_members, f_two, s_two, flush=True)
    quantiles: dict[str, np.ndarray] = {}
    quantiles["a2_mean"] = farm.mu0[:, None] + farm.offsets0
    flat = np.repeat(farm.mu[:, None], common.N_MEMBERS, axis=1)
    quantiles["a2b_rescale"] = mixture(centres=flat, offsets=farm.offsets_m, f=f_rescale)
    quantiles["a4_members"] = mixture(centres=farm.response, offsets=farm.offsets_m, f=f_members)
    two = centres_two_number(mu=farm.mu, sigma=farm.sigma, s=s_two)
    quantiles["a3_two_number"] = mixture(centres=two, offsets=farm.offsets_m, f=f_two)
    for strong, label in ((False, "shuffle"), (True, "strong")):
        source = donors(farm=farm, strong=strong, seed=SEED + 7 * band + int(strong))
        shuffled_sigma = farm.sigma[source]
        quantiles[f"p3_{label}"] = mixture(
            centres=centres_two_number(mu=farm.mu, sigma=shuffled_sigma, s=s_two),
            offsets=farm.offsets_m,
            f=f_two,
        )
        deviations = farm.response[source] - farm.mu[source][:, None]
        quantiles[f"p4_{label}"] = mixture(
            centres=farm.mu[:, None] + deviations, offsets=farm.offsets_m, f=f_members
        )
    quantiles |= heuristic_and_climatology(farm=farm, fold=fold, full=full_output())
    return [
        frame_of(farm=farm, fold=fold, quantiles=values, arm=arm)
        for arm, values in quantiles.items()
    ]


def xgboost_task(job: tuple[str, str]) -> list[pl.DataFrame]:
    """Fit XGBoost on one farm's tuning rows of all lead bands and predict its scored rows."""
    fold, site = job
    bands = [farms_of(fold, band).get(site) for band in range(len(common.BAND_SHIFT_DAYS))]
    bands = [b for b in bands if b is not None]
    if not bands:
        return []

    def features(farm: FarmBand) -> np.ndarray:
        table = farm.table.with_columns(elevation_deg=90.0 - pl.col("zenith_deg"))
        return table.select(list(XGB_FEATURES)).to_numpy().astype(np.float64)

    train_x = np.concatenate([features(b)[b.tuning] for b in bands])
    train_y = np.concatenate([b.output[b.tuning] for b in bands])
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=LEVELS,
        tree_method="hist",
        n_estimators=150,
        learning_rate=0.15,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        n_jobs=1,
        random_state=SEED,
    )
    model.fit(train_x, train_y)
    out = []
    for b in bands:
        predicted = np.asarray(model.predict(features(b))).reshape(b.table.height, len(LEVELS))
        out.append(frame_of(farm=b, fold=fold, quantiles=predicted, arm="a5_xgboost"))
    return out


def main() -> None:
    """Build all arms for both folds and write one parquet per fold."""
    n_bands = len(common.BAND_SHIFT_DAYS)
    jobs = [
        (fold, band, site)
        for fold, band in itertools.product(common.FOLDS, range(n_bands))
        for site in farms_of(fold, band)
    ]
    xjobs = [(fold, site) for fold in common.FOLDS for site in common.FARMS]
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        frames = [f for part in pool.imap_unordered(task, jobs) for f in part]
        frames += [f for part in pool.imap_unordered(xgboost_task, xjobs) for f in part]
    result = pl.concat(frames, how="diagonal")
    for fold in common.FOLDS:
        result.filter(pl.col("fold") == fold).write_parquet(common.ARMS_DIR / f"{fold}.parquet")
    print(result.group_by("fold", "arm").len().sort("fold", "arm"))


if __name__ == "__main__":
    main()
