# ruff: noqa: E501
"""Fit small XGBoost components inside the XGBoost-free metered solar-farm forecast.

Copies the farm tables, the plant forecast, and the two-number mixture of experiment 224's
`e00224_arms.py` (read from that experiment's stored member tables, 12 to 36 hour band only), then
adds, per farm and seed:

| Arm | What it adds to the plant forecast |
|---|---|
| `R` | XGBoost on the plant mean's residual; the corrected mean enters the two-number mixture |
| `S` | XGBoost with 11 quantile outputs on the residual; quantiles are the plant mean plus offsets |
| `RS` | R's corrected mean, then S's offsets fitted on R's out-of-fold residual |
| `R0` | R without the hour and the month (control for calendar memorisation) |
| `PR` | R fitted on residuals permuted within month and hour (placebo) |
| `A3r` | experiment 224's arm 3 re-run through this code with these seeds |

Tuning months (before the fold start) give out-of-fold residuals from 6 folds grouped by month, so
the offset bins never see a model's in-sample fit. Output: `predictions/<fold>.parquet`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2/e00240_fit.py
"""

import multiprocessing
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import itertools  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import norm  # noqa: E402
from studies.ens_members import MIXTURE_LEVELS, mixture_quantiles  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402
from studies.tail_calibration import level_column  # noqa: E402

SOURCE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number" / "members"
)
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00240-no-xgboost-forecast-sprinkle-part-2"
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
BAND: Final[int] = 0
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FOLDS: Final[dict[str, tuple[datetime, datetime]]] = {
    "scored_year": (datetime(2025, 9, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)),
    "replicate": (datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC)),
}
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
N_MEMBERS: Final[int] = 51
Z_POINTS: Final[np.ndarray] = norm.ppf((np.arange(N_MEMBERS) + 0.5) / N_MEMBERS)
F_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
S_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.5001, 0.1), 2)
CEILING_QUANTILE: Final[float] = 0.995
N_BINS: Final[int] = 4
MIN_BIN_ROWS: Final[int] = 50
MIN_TUNING_ROWS: Final[int] = 500
MIN_FORECAST_POWER: Final[float] = 0.02
OOF_FOLDS: Final[int] = 6
LAG: Final[timedelta] = timedelta(hours=48)
FIT_ROWS: Final[int] = 10_000
CHUNK: Final[int] = 4000
N_PROCESSES: Final[int] = 4
BASE_FEATURES: Final[tuple[str, ...]] = (
    "mu",
    "sigma",
    "ghi_mean",
    "ghi_std",
    "temperature_mean",
    "dni_mean",
    "dhi_mean",
    "elevation_deg",
    "sun_azimuth_deg",
    "lead_hours",
    "error_48h",
)
CALENDAR_FEATURES: Final[tuple[str, ...]] = ("hour", "month")
ARMS: Final[tuple[str, ...]] = ("A3r", "R", "S", "RS", "R0", "PR")


@dataclass
class FarmFold:
    """One farm's 12 to 36 hour rows with its plant forecast, for one fold."""

    site: str
    table: pl.DataFrame
    tuning: np.ndarray
    scored: np.ndarray
    output: np.ndarray
    unit: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    scale: float
    ceiling: float


def build_farm(*, site: str, fold: str) -> FarmFold | None:
    """Fit the plant scale and ceiling on the tuning rows and return the farm's arrays."""
    start, end = FOLDS[fold]
    table = pl.read_parquet(SOURCE_DIR / f"{site}_b{BAND}.parquet")
    unit = np.load(SOURCE_DIR / f"{site}_b{BAND}_unit.npy").astype(np.float64)
    time = table["time"]
    tuning = (time < start).to_numpy()
    scored = ((time >= start) & (time < end)).to_numpy()
    if tuning.sum() < MIN_TUNING_ROWS or scored.sum() == 0:
        return None
    output = table["output"].to_numpy().astype(np.float64)
    x = table["unit_mean"].to_numpy()
    scale = float(np.sum(output[tuning] * x[tuning]) / np.sum(x[tuning] ** 2))
    ceiling = float(np.quantile(output[tuning], CEILING_QUANTILE))
    response = np.minimum(scale * unit, ceiling)
    return FarmFold(
        site=site,
        table=table,
        tuning=tuning,
        scored=scored,
        output=output,
        unit=unit,
        mu=response.mean(axis=1),
        sigma=response.std(axis=1),
        scale=scale,
        ceiling=ceiling,
    )


def bin_offsets(*, forecast: np.ndarray, residual: np.ndarray, tuning: np.ndarray) -> np.ndarray:
    """Return the (rows, 11) residual quantiles of each row's forecast bin (tuning rows only)."""
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
    return table[bins]


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


def fit_factors(*, farms: list[FarmFold], seed: int) -> tuple[float, float]:
    """Return the shared `(f, s)` of experiment 224's two-number arm, fitted on other farms."""
    rng = np.random.default_rng(seed)
    pieces = [
        (farm, np.flatnonzero(farm.tuning & (farm.mu > MIN_FORECAST_POWER))) for farm in farms
    ]
    total = sum(len(rows) for _, rows in pieces)
    keep = min(FIT_ROWS, total)
    chosen = [
        (
            farm,
            np.sort(
                rng.choice(
                    rows,
                    size=min(max(1, round(keep * len(rows) / total)), len(rows)),
                    replace=False,
                )
            ),
        )
        for farm, rows in pieces
    ]
    actual = np.concatenate([f.output[r] for f, r in chosen])
    mu = np.concatenate([f.mu[r] for f, r in chosen])
    sigma = np.concatenate([f.sigma[r] for f, r in chosen])
    offsets = np.concatenate(
        [
            bin_offsets(forecast=f.mu, residual=f.output - f.mu, tuning=f.tuning)[r]
            for f, r in chosen
        ]
    )

    def loss(f: float, s: float) -> float:
        quantiles = mixture(
            centres=centres_two_number(mu=mu, sigma=sigma, s=s), offsets=offsets, f=f
        )
        return pinball(actual=actual, quantiles=np.maximum(quantiles, 0.0))

    f, s = 1.0, 1.0
    f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
    s = float(S_GRID[int(np.argmin([loss(f, g) for g in S_GRID]))])
    f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
    return f, s


def feature_matrix(*, farm: FarmFold, columns: tuple[str, ...], base: np.ndarray) -> np.ndarray:
    """Return the model inputs: plant forecast, weather, sun, lead, the error 48 hours earlier."""
    table = farm.table.with_columns(
        mu=pl.Series(base),
        sigma=pl.Series(farm.sigma),
        elevation_deg=90.0 - pl.col("zenith_deg"),
    )
    lagged = table.select(
        time=pl.col("time") + LAG, error_48h=pl.col("output").cast(pl.Float64) - pl.col("mu")
    )
    table = table.join(lagged, on="time", how="left").with_columns(
        hour=pl.col("time").dt.hour(), month=pl.col("time").dt.month()
    )
    return table.select(list(columns)).to_numpy().astype(np.float64)


def oof_groups(*, farm: FarmFold) -> np.ndarray:
    """Return each tuning row's out-of-fold group: a block of consecutive months.

    Consecutive blocks keep most calendar months in every training set, so the month input is not
    penalised in the out-of-fold residuals that build the offsets.
    """
    months = farm.table["time"].dt.strftime("%Y-%m")
    unique = sorted(set(months.to_list()))
    block = int(np.ceil(len(unique) / OOF_FOLDS))
    index = {m: i // block for i, m in enumerate(unique)}
    return np.array([index[m] for m in months.to_list()])


def regressor(*, seed: int, quantile: bool) -> xgb.XGBRegressor:
    """Return a small XGBoost model with no column subsampling."""
    params: dict[str, object] = {
        "tree_method": "hist",
        "n_estimators": 150,
        "learning_rate": 0.15,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "n_jobs": 1,
        "random_state": seed,
    }
    if quantile:
        params |= {"objective": "reg:quantileerror", "quantile_alpha": LEVELS}
    return xgb.XGBRegressor(**params)


def residual_correction(
    *, farm: FarmFold, columns: tuple[str, ...], seed: int, permute: bool
) -> np.ndarray:
    """Return a correction to the plant mean for every row: out-of-fold on tuning, full-fit scored."""
    x = feature_matrix(farm=farm, columns=columns, base=farm.mu)
    y = farm.output - farm.mu
    rng = np.random.default_rng(seed)
    if permute:
        keys = (
            farm.table["time"].dt.strftime("%Y-%m")
            + "|"
            + farm.table["time"].dt.hour().cast(pl.String)
        ).to_numpy()
        y = y.copy()
        for key in np.unique(keys):
            rows = np.flatnonzero((keys == key) & farm.tuning)
            y[rows] = y[rng.permutation(rows)]
    groups = oof_groups(farm=farm)
    delta = np.zeros(len(y))
    for group in range(OOF_FOLDS):
        train = farm.tuning & (groups != group)
        held = farm.tuning & (groups == group)
        if held.sum() == 0:
            continue
        model = regressor(seed=seed, quantile=False).fit(x[train], y[train])
        delta[held] = model.predict(x[held])
    final = regressor(seed=seed, quantile=False).fit(x[farm.tuning], y[farm.tuning])
    delta[~farm.tuning] = final.predict(x[~farm.tuning])
    return delta


def quantile_offsets(
    *, farm: FarmFold, columns: tuple[str, ...], base: np.ndarray, seed: int
) -> np.ndarray:
    """Return XGBoost's 11 quantile offsets of the residual from `base`, for every row."""
    x = feature_matrix(farm=farm, columns=columns, base=base)
    y = farm.output - base
    model = regressor(seed=seed, quantile=True).fit(x[farm.tuning], y[farm.tuning])
    return np.asarray(model.predict(x)).reshape(len(y), len(LEVELS))


def frame_of(
    *, farm: FarmFold, fold: str, quantiles: np.ndarray, arm: str, seed: int
) -> pl.DataFrame:
    """Return the scored rows of one arm in the layout of experiment 224's predictions."""
    rows = farm.scored
    clipped = np.maximum(np.sort(quantiles[rows].astype(np.float64), axis=1), 0.0)
    return (
        farm.table.filter(pl.Series(rows))
        .select(
            time_series_id="site",
            power_fcst_init_time="init_time",
            valid_time="time",
            lead_hours="lead_hours",
            band="band",
            actual="output",
        )
        .with_columns(
            seed=pl.lit(seed, dtype=pl.Int32),
            actual=pl.col("actual").cast(pl.Float64),
            lead_hours=pl.col("lead_hours").cast(pl.Float64),
            fold=pl.lit(fold),
            arm=pl.lit(arm),
            **{level_column(level=lv): pl.Series(clipped[:, i]) for i, lv in enumerate(LEVELS)},
        )
    )


def task(job: tuple[str, str, int]) -> list[pl.DataFrame]:
    """Build every arm for one farm, fold, and seed."""
    fold, site, seed = job
    farms = {s: build_farm(site=s, fold=fold) for s in FARMS}
    farm = farms[site]
    if farm is None:
        return []
    others = [f for s, f in farms.items() if s != site and f is not None]
    f_two, s_two = fit_factors(farms=others, seed=240 + seed)
    full = (*BASE_FEATURES, *CALENDAR_FEATURES)
    out: dict[str, np.ndarray] = {}

    def corrected_mixture(*, centre: np.ndarray) -> np.ndarray:
        offsets = bin_offsets(forecast=centre, residual=farm.output - centre, tuning=farm.tuning)
        return mixture(
            centres=centres_two_number(mu=centre, sigma=farm.sigma, s=s_two),
            offsets=offsets,
            f=f_two,
        )

    out["A3r"] = corrected_mixture(centre=farm.mu)
    for arm, columns, permute in (
        ("R", full, False),
        ("R0", BASE_FEATURES, False),
        ("PR", full, True),
    ):
        delta = residual_correction(farm=farm, columns=columns, seed=seed, permute=permute)
        centre = np.clip(farm.mu + delta, 0.0, farm.ceiling)
        out[arm] = corrected_mixture(centre=centre)
        if arm == "R":
            centre_r = centre
    out["S"] = farm.mu[:, None] + quantile_offsets(farm=farm, columns=full, base=farm.mu, seed=seed)
    out["RS"] = centre_r[:, None] + quantile_offsets(
        farm=farm, columns=full, base=centre_r, seed=seed
    )
    print("done", fold, site, seed, flush=True)
    return [
        frame_of(farm=farm, fold=fold, quantiles=q, arm=arm, seed=seed) for arm, q in out.items()
    ]


def main() -> None:
    """Build every arm for both folds and write one parquet per fold."""
    jobs = list(itertools.product(FOLDS, FARMS, SEEDS))
    (OUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        frames = [f for part in pool.imap_unordered(task, jobs) for f in part]
    result = pl.concat(frames, how="diagonal")
    for fold in FOLDS:
        path = OUT_DIR / "predictions" / f"{fold}.parquet"
        result.filter(pl.col("fold") == fold).write_parquet(path)
    print(result.group_by("fold", "arm").len().sort("fold", "arm"))


if __name__ == "__main__":
    main()
