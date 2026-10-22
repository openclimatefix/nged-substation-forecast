"""Build every arm's 11 quantiles for each farm, fold, and lead band, and write them.

All arms predict power over capacity from the fleet-shape plant of experiments 218 and 224.

Arms, with the weather each uses and how each builds its centre and spread:

- `a1_ens_two_number`: ensemble weather. Centre: mean of 51 plant responses. Spread: 51 Gaussian
  points of their standard deviation, times the offsets.
- `a1x_tree_ens`: ensemble weather. XGBoost on ensemble columns, 11 quantile outputs.
- `a2_ukv_alone`: UKV weather. Plant on UKV shortwave and temperature, binned residual quantiles.
- `a3_blend`: both. Convex blend of the two plant forecasts by UKV cloud band, ensemble spread.
- `a4_tree_all_ukv`: both. XGBoost on ensemble columns and 22 UKV variables.
- `a5_tree_ukv_sky`: both. The same tree with UKV shortwave and cloud only.
- `p3_blend_shuffle`, `p4_tree_shuffle`: arms 3 and 4 with UKV from a donor row (same month, hour
  band, and cloud band).

The tree predicts output over clear-sky irradiance (divided by 1000), then multiplies back.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00229-*/e00229_arms.py
"""

import itertools
import multiprocessing
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import Final

import e00229_common as common
import numpy as np
import polars as pl
import pvlib
import xgboost as xgb
from baseline_forecasters.climatology import CLIMATOLOGY_POOLING_OFFSETS
from scipy.stats import norm
from studies.ens_members import mixture_quantiles
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.tail_calibration import level_column

N_PROCESSES: Final[int] = 4
LEVELS: Final[np.ndarray] = np.array(common.LEVELS)
Z_POINTS: Final[np.ndarray] = norm.ppf((np.arange(common.N_MEMBERS) + 0.5) / common.N_MEMBERS)
F_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
S_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.5001, 0.1), 2)
W_GRID: Final[np.ndarray] = np.round(np.arange(0.0, 1.0001, 0.1), 2)
FIT_ROWS: Final[int] = 10_000
CHUNK: Final[int] = 4000
N_BINS: Final[int] = 4
MIN_BIN_ROWS: Final[int] = 50
MIN_TUNING_ROWS: Final[int] = 500
MIN_FORECAST_POWER: Final[float] = 0.02
SEED: Final[int] = 229
CLOUD_EDGES: Final[tuple[float, float]] = (30.0, 70.0)
"""UKV total cloud cover (per cent) below which a row is clear and above which it is overcast."""
HEURISTIC_LAG_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 49, 50, 51, 52, 53, 54, 55)
HEAT_PER_KELVIN: Final[float] = 0.004
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
CEILING_QUANTILE: Final[float] = common.CEILING_QUANTILE
ENS_COLUMNS: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "temperature_mean",
    "dni_mean",
    "dhi_mean",
    "unit_mean",
    "mu_ens",
    "sigma_ens",
    "elevation_deg",
    "sun_azimuth_deg",
    "lead_hours",
    "clear_sky",
)
UKV_ALL: Final[tuple[str, ...]] = (
    *(f"ukv_{v}" for v in common.UKV_VARIABLES if v != "wind_direction_10m"),
    "ukv_wind_u",
    "ukv_wind_v",
    "ukv_unit",
    "ukv_mu",
)
UKV_SKY: Final[tuple[str, ...]] = (
    *(f"ukv_{v}" for v in common.UKV_IRRADIANCE_AND_CLOUD),
    "ukv_unit",
    "ukv_mu",
)
TREE_FLOOR_IRRADIANCE: Final[float] = 0.05
TREE_RATIO_CAP: Final[float] = 3.0
ARM_NAMES: Final[tuple[str, ...]] = (
    "floor_heuristic",
    "floor_climatology",
    "a1_ens_two_number",
    "a1x_tree_ens",
    "a2_ukv_alone",
    "a3_blend",
    "a4_tree_all_ukv",
    "a5_tree_ukv_sky",
    "p3_blend_shuffle",
    "p4_tree_shuffle",
)


def unit_response(
    *,
    ghi: np.ndarray,
    temperature_c: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
    doy: np.ndarray,
) -> np.ndarray:
    """Return the fleet-shape unit response (power over capacity per unit scale) per row."""
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=doy)
    sky = SunAndSky(
        half_hour_end_time=np.zeros(ghi.size),
        ghi_w_m2=ghi,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"])),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"])),
        zenith_deg=zenith,
        sun_azimuth_deg=azimuth,
    )
    poa = plane_of_array_w_m2(
        sky=sky, tilt_deg=FLEET_TILT_DEG, azimuth_deg=FLEET_AZIMUTH_DEG, tracker=False
    )
    unit = poa / 1000.0
    cell = temperature_c + CELL_TEMPERATURE_RISE_K * unit
    derating = np.clip(1.0 - HEAT_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0.0, None)
    return unit * derating


@dataclass
class FarmBand:
    """One farm's rows for one lead band and fold, with the arrays every arm needs."""

    site: str
    band: int
    table: pl.DataFrame
    tuning: np.ndarray
    scored: np.ndarray
    output: np.ndarray
    cloud_band: np.ndarray
    mu_ens: np.ndarray
    sigma: np.ndarray
    response: np.ndarray
    mu_ukv: np.ndarray
    offsets_ens: np.ndarray
    offsets_ukv: np.ndarray
    bin_ens: np.ndarray
    bin_ukv: np.ndarray
    ceiling: float
    scale_ens: float
    scale_ukv: float


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


@cache
def clear_sky_table() -> pl.DataFrame:
    """Return the clear-sky irradiance per farm and time (W/m²)."""
    return pl.read_parquet(common.FARM_ROWS).select("site", "time", "clear_sky")


def load_table(*, site: str, band: int) -> pl.DataFrame:
    """Join a farm and band's ensemble table to its UKV table, UKV unit response and clear-sky."""
    members = pl.read_parquet(common.members_table(site=site, band=band))
    ukv = pl.read_parquet(common.UKV_DIR / f"{site}_b{band}.parquet").drop("site", "band")
    table = members.join(ukv, on=["time", "init_time"], how="inner").join(
        clear_sky_table().filter(pl.col("site") == site).drop("site"), on="time", how="left"
    )
    lo, hi = common.BAND_LEAD_RANGE[band]
    table = table.filter(
        pl.col("ukv_usable")
        & pl.col("ukv_shortwave_down").is_finite()
        & pl.col("ukv_temperature_1p5m").is_finite()
        & pl.col("clear_sky").is_not_null()
        & (pl.col("lead_hours") >= lo)
        & (pl.col("lead_hours") <= hi)
    )
    ghi = table["ukv_shortwave_down"].to_numpy()
    unit = unit_response(
        ghi=ghi,
        temperature_c=table["ukv_temperature_1p5m"].to_numpy() - 273.15,
        zenith=table["zenith_deg"].to_numpy(),
        azimuth=table["sun_azimuth_deg"].to_numpy(),
        doy=table["time"].dt.ordinal_day().to_numpy(),
    )
    return table.with_columns(ukv_unit=pl.Series(unit), elevation_deg=90.0 - pl.col("zenith_deg"))


def build_farm_band(*, site: str, band: int, fold: str) -> FarmBand | None:
    """Fit the plant scales and ceiling on the tuning rows and return the farm-band arrays."""
    start, end = common.FOLDS[fold]
    table = load_table(site=site, band=band)
    # Keep the ensemble member responses aligned with the surviving rows.
    full = pl.read_parquet(
        common.members_table(site=site, band=band), columns=["time", "init_time"]
    )
    unit_all = np.load(common.members_unit(site=site, band=band)).astype(np.float64)
    keep = (
        full.with_row_index("i")
        .join(table.select("time", "init_time"), on=["time", "init_time"], how="inner")["i"]
        .to_numpy()
    )
    unit = unit_all[keep]
    if not (full["time"].to_numpy()[keep] == table["time"].to_numpy()).all():
        msg = "ensemble member responses are not aligned with the joined table"
        raise ValueError(msg)
    time = table["time"]
    tuning = (time < start).to_numpy()
    scored = ((time >= start) & (time < end)).to_numpy()
    if tuning.sum() < MIN_TUNING_ROWS or scored.sum() == 0:
        return None
    output = table["output"].to_numpy().astype(np.float64)
    x_ens = table["unit_mean"].to_numpy()
    x_ukv = table["ukv_unit"].to_numpy()
    scale_ens = float(np.sum(output[tuning] * x_ens[tuning]) / np.sum(x_ens[tuning] ** 2))
    scale_ukv = float(np.sum(output[tuning] * x_ukv[tuning]) / np.sum(x_ukv[tuning] ** 2))
    ceiling = float(np.quantile(output[tuning], CEILING_QUANTILE))
    response = np.minimum(scale_ens * unit, ceiling)
    mu_ens = response.mean(axis=1)
    sigma = response.std(axis=1)
    mu_ukv = np.minimum(scale_ukv * x_ukv, ceiling)
    cloud = table["ukv_cloud_total"].to_numpy()
    cloud_band = np.digitize(np.nan_to_num(cloud, nan=50.0), CLOUD_EDGES)
    bin_ens, offsets_ens = _bin_offsets(forecast=mu_ens, residual=output - mu_ens, tuning=tuning)
    bin_ukv, offsets_ukv = _bin_offsets(forecast=mu_ukv, residual=output - mu_ukv, tuning=tuning)
    table = table.with_columns(
        mu_ens=pl.Series(mu_ens), sigma_ens=pl.Series(sigma), ukv_mu=pl.Series(mu_ukv)
    )
    return FarmBand(
        site=site,
        band=band,
        table=table,
        tuning=tuning,
        scored=scored,
        output=output,
        cloud_band=cloud_band,
        mu_ens=mu_ens,
        sigma=sigma,
        response=response,
        mu_ukv=mu_ukv,
        offsets_ens=offsets_ens,
        offsets_ukv=offsets_ukv,
        bin_ens=bin_ens,
        bin_ukv=bin_ukv,
        ceiling=ceiling,
        scale_ens=scale_ens,
        scale_ukv=scale_ukv,
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
    *, pool: dict[str, np.ndarray], mu_key: str, offsets_key: str
) -> tuple[float, float]:
    """Return `(f, s)` minimising the pinball loss on the pooled tuning rows."""

    def loss(f: float, s: float) -> float:
        centres = centres_two_number(mu=pool[mu_key], sigma=pool["sigma"], s=s)
        quantiles = mixture(centres=centres, offsets=pool[offsets_key], f=f)
        return pinball(actual=pool["actual"], quantiles=np.maximum(quantiles, 0.0))

    f, s = 1.0, 1.0
    f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
    s = float(S_GRID[int(np.argmin([loss(f, g) for g in S_GRID]))])
    f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
    return f, s


def blend_weights(*, farms: list[FarmBand]) -> np.ndarray:
    """Return the ensemble's weight for each UKV cloud band, fitted on the farms' tuning rows.

    The two plant forecasts are combined convexly, so `w` runs from 0 (UKV alone) to 1 (ensemble
    alone); each weight minimises the blended forecast's squared error over a cloud band's rows.
    """
    weights = np.full(len(CLOUD_EDGES) + 1, 0.5)
    for cloud in range(len(weights)):
        parts = [
            (f.mu_ens[r], f.mu_ukv[r], f.output[r])
            for f in farms
            for r in [f.tuning & (f.cloud_band == cloud) & (f.mu_ens > MIN_FORECAST_POWER)]
        ]
        ens, ukv, actual = (np.concatenate(p) for p in zip(*parts, strict=True))
        if len(actual) < MIN_BIN_ROWS:
            continue
        errors = [np.mean((actual - (w * ens + (1 - w) * ukv)) ** 2) for w in W_GRID]
        weights[cloud] = float(W_GRID[int(np.argmin(errors))])
    return weights


def make_pool(
    *, farms: list[FarmBand], seed: int, mu_attr: str, offsets_attr: str
) -> dict[str, np.ndarray]:
    """Return up to `FIT_ROWS` tuning daylight rows pooled over the given farms."""
    rng = np.random.default_rng(seed)
    pieces = [
        (farm, np.flatnonzero(farm.tuning & (getattr(farm, mu_attr) > MIN_FORECAST_POWER)))
        for farm in farms
    ]
    total = sum(len(rows) for _, rows in pieces)
    keep = min(FIT_ROWS, total)
    chosen = []
    for farm, rows in pieces:
        take = max(1, round(keep * len(rows) / total))
        chosen.append((farm, np.sort(rng.choice(rows, size=min(take, len(rows)), replace=False))))
    return {
        "actual": np.concatenate([f.output[r] for f, r in chosen]),
        "mu": np.concatenate([getattr(f, mu_attr)[r] for f, r in chosen]),
        "sigma": np.concatenate([f.sigma[r] for f, r in chosen]),
        "offsets": np.concatenate([getattr(f, offsets_attr)[r] for f, r in chosen]),
    }


def donors(*, farm: FarmBand, seed: int) -> np.ndarray:
    """Return a donor row per scored row: a random row of its month, hour band, and cloud band."""
    table = farm.table
    key = (
        table["time"].dt.month().cast(pl.String)
        + "|"
        + (table["time"].dt.hour() // 3).cast(pl.String)
        + "|"
        + pl.Series(farm.cloud_band.astype(str))
    ).to_numpy()
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
        clear_sky="clear_sky",
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
    return pl.read_parquet(common.FARM_ROWS).select("site", "time", "output")


def task(job: tuple[str, int, str]) -> list[pl.DataFrame]:
    """Fit shared factors and blend weights leave-one-farm-out; build one farm-band's arms."""
    fold, band, site = job
    farms = farms_of(fold, band)
    farm = farms[site]
    others = [f for name, f in farms.items() if name != site]
    pool_ens = make_pool(
        farms=others, seed=SEED + band, mu_attr="mu_ens", offsets_attr="offsets_ens"
    )
    f1, s1 = fit_factors(pool=pool_ens, mu_key="mu", offsets_key="offsets")
    weights = blend_weights(farms=others)

    def blended(*, mu_ukv: np.ndarray, cloud_band: np.ndarray) -> np.ndarray:
        w = weights[cloud_band]
        return w * farm.mu_ens + (1.0 - w) * mu_ukv

    mu_blend = blended(mu_ukv=farm.mu_ukv, cloud_band=farm.cloud_band)
    _, offsets_b = _bin_offsets(
        forecast=mu_blend, residual=farm.output - mu_blend, tuning=farm.tuning
    )
    # Offsets for the blend come from the blended forecast's own tuning residuals.
    pool_b = {
        "actual": np.concatenate(
            [f.output[f.tuning & (f.mu_ens > MIN_FORECAST_POWER)] for f in others]
        ),
        "mu": np.concatenate(
            [
                (weights[f.cloud_band] * f.mu_ens + (1 - weights[f.cloud_band]) * f.mu_ukv)[
                    f.tuning & (f.mu_ens > MIN_FORECAST_POWER)
                ]
                for f in others
            ]
        ),
        "sigma": np.concatenate(
            [f.sigma[f.tuning & (f.mu_ens > MIN_FORECAST_POWER)] for f in others]
        ),
    }
    offsets_pool = []
    for f in others:
        m = weights[f.cloud_band] * f.mu_ens + (1 - weights[f.cloud_band]) * f.mu_ukv
        _, off = _bin_offsets(forecast=m, residual=f.output - m, tuning=f.tuning)
        offsets_pool.append(off[f.tuning & (f.mu_ens > MIN_FORECAST_POWER)])
    pool_b["offsets"] = np.concatenate(offsets_pool)
    if len(pool_b["actual"]) > FIT_ROWS:
        rng = np.random.default_rng(SEED + 100 + band)
        pick = np.sort(rng.choice(len(pool_b["actual"]), size=FIT_ROWS, replace=False))
        pool_b = {k: v[pick] for k, v in pool_b.items()}
    f3, s3 = fit_factors(pool=pool_b, mu_key="mu", offsets_key="offsets")
    print(
        "factors", fold, band, site, "ens", f1, s1, "blend", f3, s3, "weights", weights, flush=True
    )
    quantiles: dict[str, np.ndarray] = {}
    two = centres_two_number(mu=farm.mu_ens, sigma=farm.sigma, s=s1)
    quantiles["a1_ens_two_number"] = mixture(centres=two, offsets=farm.offsets_ens, f=f1)
    quantiles["a2_ukv_alone"] = farm.mu_ukv[:, None] + farm.offsets_ukv
    two_b = centres_two_number(mu=mu_blend, sigma=farm.sigma, s=s3)
    quantiles["a3_blend"] = mixture(centres=two_b, offsets=offsets_b, f=f3)
    source = donors(farm=farm, seed=SEED + 7 * band)
    mu_shuffled = blended(mu_ukv=farm.mu_ukv[source], cloud_band=farm.cloud_band)
    two_p = centres_two_number(mu=mu_shuffled, sigma=farm.sigma, s=s3)
    quantiles["p3_blend_shuffle"] = mixture(centres=two_p, offsets=offsets_b, f=f3)
    quantiles |= heuristic_and_climatology(farm=farm, fold=fold, full=full_output())
    return [
        frame_of(farm=farm, fold=fold, quantiles=values, arm=arm)
        for arm, values in quantiles.items()
    ]


def tree_matrix(
    *, farm: FarmBand, columns: tuple[str, ...], source: np.ndarray | None = None
) -> np.ndarray:
    """Return the tree's feature matrix; `source` swaps in the donor rows' UKV columns."""
    table = farm.table
    ens_names = [c for c in columns if not c.startswith("ukv_")]
    ukv_names = [c for c in columns if c.startswith("ukv_")]
    parts = [table.select(ens_names).to_numpy().astype(np.float64)]
    if ukv_names:
        ukv = table.select(ukv_names).to_numpy().astype(np.float64)
        parts.append(ukv if source is None else ukv[source])
    return np.concatenate(parts, axis=1)


def ratio_target(*, farm: FarmBand) -> tuple[np.ndarray, np.ndarray]:
    """Return the tree's target (output over clear-sky irradiance / 1000) and its divisor."""
    divisor = np.maximum(farm.table["clear_sky"].to_numpy() / 1000.0, TREE_FLOOR_IRRADIANCE)
    return np.minimum(farm.output / divisor, TREE_RATIO_CAP), divisor


def xgboost_task(job: tuple[str, str]) -> list[pl.DataFrame]:
    """Fit the three trees on one farm's tuning rows of both bands and predict its scored rows."""
    fold, site = job
    bands = [farms_of(fold, band).get(site) for band in common.BANDS]
    bands = [b for b in bands if b is not None]
    if not bands:
        return []
    arms = {
        "a1x_tree_ens": ENS_COLUMNS,
        "a4_tree_all_ukv": (*ENS_COLUMNS, *UKV_ALL),
        "a5_tree_ukv_sky": (*ENS_COLUMNS, *UKV_SKY),
    }
    out = []
    for arm, columns in arms.items():
        train_x = np.concatenate([tree_matrix(farm=b, columns=columns)[b.tuning] for b in bands])
        train_y = np.concatenate([ratio_target(farm=b)[0][b.tuning] for b in bands])
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
        for b in bands:
            divisor = ratio_target(farm=b)[1]
            predicted = np.asarray(model.predict(tree_matrix(farm=b, columns=columns))).reshape(
                b.table.height, len(LEVELS)
            )
            out.append(frame_of(farm=b, fold=fold, quantiles=predicted * divisor[:, None], arm=arm))
            if arm == "a4_tree_all_ukv":
                source = donors(farm=b, seed=SEED + 31 * b.band)
                shuffled = np.asarray(
                    model.predict(tree_matrix(farm=b, columns=columns, source=source))
                ).reshape(b.table.height, len(LEVELS))
                out.append(
                    frame_of(
                        farm=b,
                        fold=fold,
                        quantiles=shuffled * divisor[:, None],
                        arm="p4_tree_shuffle",
                    )
                )
    return out


def main() -> None:
    """Build all arms for both folds and write one parquet per fold."""
    jobs = [
        (fold, band, site)
        for fold, band in itertools.product(common.FOLDS, common.BANDS)
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
