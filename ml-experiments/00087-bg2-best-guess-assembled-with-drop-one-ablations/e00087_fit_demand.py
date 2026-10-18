"""Fit the demand members of experiment 00087 and write each refit's 11 quantiles per daily run.

Copied from experiment 00048's `e00048_fit.py` (the per-refit parallel fitter), with the weekly
analogue columns of experiment 00034's `e00034_fit.py` and `e00034_lags.py` (`a7`), the pooled-by-
type fit of experiment 00049's `e00049_fit.py` (`c0b`), and the offset arms of experiment 00074's
`e00074_fit.py` (`e00087_offset.py` holds the offset machinery). Widened to 11 quantile levels.

One job is one refit of one configuration for a chunk of series (or, for the pooled configuration,
one pool). A refit at time `t` trains each series' N1 model on the ENS control member's rows whose
valid time is at least 3.5 hours before `t` and whose run began at or before `t`, weighted by a
recency half-life anchored at `t`. It forecasts the daily runs that the configuration's regimes
assign to it (`e00087_regimes`), from the 365 daily runs of 2025-07-01 to 2026-06-30, and writes
the mixture of the 51 ensemble members' 11 quantiles to
`work/anchor_preds/<config>[_o|_oheld]/<site>_s<seed>_<refit>.parquet`, keyed by series, run, and
valid time. Existing files stay, so a stopped run resumes.

Configurations, each a member of the BG2 blend:

- `n1h`: the weekly (W) and monthly (M) regimes, half-life 0.5 years, 500 rounds, `eta` 0.05,
  Primary, BSP, and GSP. The refit 2025-07-01 is shared by the two regimes, and its file holds both
  regimes' runs.
- `frozen`: the stack's N1 (ST), trained once on rows to 2025-06-30 with a half-life of 2 years
  anchored at 2025-06-30, 500 rounds, forecasting all 365 runs. Primary, BSP, and GSP.
- `an`: the monthly regime with experiment 00030's weekly-analogue mean and median columns (AN).
  Primary.
- `pt`: one XGBoost model per metered-units pool (MW or MVA) of Primary series, the series
  identifier as a categorical column, the monthly regime, whole history, half-life 0.5 years, 500
  rounds (PT). Interpretation of experiment 00049's pooled arm `c0b`, which used P-roll.

`--offset` fits the offset twin of a configuration at the five offset Primaries (`e00087_offset`):
the base margin is the median of the refit's training labels plus the solar offset. It writes
the vintage-capacity predictions to `<config>_o` and the held-capacity predictions (ablation A6) to
`<config>_oheld`. Arms without `--offset` have no base margin, so XGBoost estimates its own start.

Leakage checks, each raising `ValueError`: every lag is no younger than the newest usable
observation (`e00087_common.origin_lags`); a refit's training rows have valid time at most 3.5
hours before the refit and a run that began at or before the refit (`known_rows`); the vintage
capacity of a run comes from weeks older than the run's week (`e00087_offset`).

The jobs run as parallel worker processes, each with `cores // workers` threads, sharing one GPU.
Every job appends its wall-clock seconds to `work/gpu_hours.jsonl`.

    uv run python ml-experiments/00087-*/e00087_fit_demand.py --config n1h --regime weekly
        [--groups Primary BSP GSP] [--offset] [--workers 2] [--cores 12] [--smoke]
    uv run python ml-experiments/00087-*/e00087_fit_demand.py --config n1h --dry-gpu-hours
"""

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common
import e00087_offset as offsets
import e00087_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
WEEK_STEPS: Final[int] = 7 * common.STEPS_PER_DAY
WEEKLY_ANALOGUE_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
A7_COLUMNS: Final[tuple[str, ...]] = ("weekly_mean", "weekly_median")
SMOKE_ROUNDS: Final[int] = 20
SMOKE_RUNS: Final[int] = 3
SMOKE_REFITS: Final[int] = 2
REGIME_NAMES: Final[tuple[str, ...]] = ("weekly", "monthly")


@dataclass(frozen=True)
class Config:
    """A training configuration.

    Attributes:
        name: The folder name under `anchor_preds`.
        half_life_days: Recency half-life of the row weights, anchored at the refit time.
        rounds: Boosting rounds.
        eta: Learning rate.
        regime_names: The regimes (keys of `regimes.REGIMES`) whose refits the configuration fits.
        groups: The substation groups the configuration is fitted for.
        frozen: Train once on the pilot's control-member rows to 2025-06-30 only (experiment
            00033's recipe), with no rows from later runs.
        a7: Add experiment 00030's weekly-analogue mean and median as input columns.
        pooled: Fit one model per metered-units pool, with the series identifier as a category.
    """

    name: str
    half_life_days: float
    rounds: int
    eta: float
    regime_names: tuple[str, ...]
    groups: tuple[str, ...]
    frozen: bool = False
    a7: bool = False
    pooled: bool = False


CONFIGS: Final[dict[str, Config]] = {
    "n1h": Config(
        name="n1h",
        half_life_days=common.SHORT_HALF_LIFE_DAYS,
        rounds=common.N_ROUNDS,
        eta=0.05,
        regime_names=("weekly", "monthly"),
        groups=common.GROUPS,
    ),
    "frozen": Config(
        name="frozen",
        half_life_days=common.RECENCY_HALF_LIFE_DAYS,
        rounds=common.N_ROUNDS,
        eta=0.05,
        regime_names=("frozen",),
        groups=common.GROUPS,
        frozen=True,
    ),
    "an": Config(
        name="an",
        half_life_days=common.SHORT_HALF_LIFE_DAYS,
        rounds=common.N_ROUNDS,
        eta=0.05,
        regime_names=("monthly",),
        groups=("Primary",),
        a7=True,
    ),
    "pt": Config(
        name="pt",
        half_life_days=common.SHORT_HALF_LIFE_DAYS,
        rounds=common.N_ROUNDS,
        eta=0.05,
        regime_names=("monthly",),
        groups=("Primary",),
        pooled=True,
    ),
}


def input_columns(*, a7: bool) -> list[str]:
    """Return the input columns in matrix order."""
    return [
        *common.feature_columns(),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
        *(A7_COLUMNS if a7 else ()),
    ]


def a7_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the mean and median of the six weekly analogues usable at the origin.

    Copied from experiment 00034's `e00034_lags.py`.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 2): `weekly_mean` and `weekly_median`. A weekly analogue is
        power 1 to 6 whole weeks before the valid time, and NaN when that half-hour is later than
        the newest usable observation.

    Raises:
        ValueError: If a case's valid time is not after its origin.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - common.NEWEST_AGE
    analogues = []
    for weeks in WEEKLY_ANALOGUE_WEEKS:
        index = target - weeks * WEEK_STEPS
        analogues.append(
            np.where(index <= newest, common.take(values=power, index=index), np.nan).astype(
                np.float32
            )
        )
    weekly = np.stack(analogues, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(weekly, axis=1), np.nanmedian(weekly, axis=1)], axis=1).astype(
            np.float32
        )


def add_lags(*, frame: pl.DataFrame, power: np.ndarray, a7: bool) -> pl.DataFrame:
    """Add the three origin lags (and the weekly-analogue columns if `a7`) to a frame of runs."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    out = frame.with_columns(
        pl.DataFrame(
            common.origin_lags(power=power, target=target, origin=origin),
            schema=list(common.LAG_COLUMNS),
            orient="row",
        ).cast(pl.Float32)
    )
    if a7:
        out = out.with_columns(
            pl.DataFrame(
                a7_lags(power=power, target=target, origin=origin),
                schema=list(A7_COLUMNS),
                orient="row",
            ).cast(pl.Float32)
        )
    return out


def training_rows(
    *, row: dict, arrays: common.SeriesArrays, frozen: bool, a7: bool
) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns and lags.

    The rows run to 2025-06-30 for the frozen configuration and to 2026-06-29 otherwise. The power
    of the rows after 2025-06-30 is filled from the dense grid. Every row's eligibility for a refit
    (valid time and run start) is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        frozen: Whether to keep only the pilot's training rows, which end at 2025-06-30.
        a7: Whether to add the weekly-analogue columns.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW),
        the weather columns, the derived columns, `lead_hours`, and the lags.
    """
    series_id = row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *WEATHER)
        .collect()
    )
    if frozen:
        rows = before
    else:
        after = pl.concat(
            [
                pl.read_parquet(path).filter(pl.col("time_series_id") == series_id)
                for path in sorted(common.AFTER_DIR.glob("init=*.parquet"))
            ]
        )
        padded = after.with_columns(power=pl.lit(None, dtype=pl.Float32)).select(before.columns)
        rows = pl.concat([before, padded.cast(before.schema)])
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    filled = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
    rows = rows.with_columns(
        power=pl.when(pl.col("power").is_null()).then(pl.Series(filled)).otherwise(pl.col("power"))
    )
    rows = rows.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=rows, power=arrays.power, a7=a7)


def known_rows(*, rows: pl.DataFrame, refit: datetime, config: Config) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see, and check that no row comes from its future.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit and its run
    began at or before the refit. The frozen model, anchored at 2025-06-30 23:30, sees the pilot's
    training rows up to that valid time (experiment 00033's recipe has no 3.5-hour gap).

    Raises:
        ValueError: If no row is usable, or a kept row is newer than the refit allows.
    """
    latest_valid = common.LAST_TRAINING_VALID if config.frozen else refit - regimes.REFIT_CUTOFF_AGE
    known = rows.filter(
        (pl.col("valid_time") <= latest_valid) & (pl.col("power_fcst_init_time") <= refit)
    )
    too_new = known.filter(
        (pl.col("valid_time") > latest_valid) | (pl.col("power_fcst_init_time") > refit)
    ).height
    if known.is_empty() or too_new:
        msg = f"refit {refit}: no usable rows or a row from after the refit"
        raise ValueError(msg)
    return known


def parameters(*, config: Config, seed: int, nthread: int) -> dict[str, object]:
    """Return the XGBoost parameters of a configuration."""
    return {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": config.eta, "nthread": nthread}


def mixture_frame(*, keys: pl.DataFrame, raw: np.ndarray, members: int, seed: int) -> pl.DataFrame:
    """Return one run's keys with the mixture of the members' 11 quantiles and the seed."""
    cube = raw.reshape(-1, members, len(common.LEVELS))
    return keys.hstack(
        pl.DataFrame(
            common.mixture_quantiles(member_quantiles=cube), schema=list(common.LEVEL_COLUMNS)
        )
    ).with_columns(seed=pl.lit(seed).cast(pl.Int8))


def job_path(*, root: Path, config: str, site: str, seed: int, refit: datetime) -> Path:
    """Return the file one refit's predictions for one series are written to."""
    return root / config / f"{site}_s{seed}_{refit:%Y%m%dT%H%M}.parquet"


_CACHE: dict[str, object] = {}


def _series_and_arrays() -> tuple[pl.DataFrame, dict[str, common.SeriesArrays]]:
    """Return the series table and arrays, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table()
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        _CACHE["rows"] = {}
        _CACHE["anomaly"] = {}
    return _CACHE["series"], _CACHE["arrays"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(
    *, row: dict, arrays: common.SeriesArrays, config: Config
) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[tuple[str, bool, bool], pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    key = (row["site"], config.frozen, config.a7)
    if key not in cache:
        cache[key] = training_rows(row=row, arrays=arrays, frozen=config.frozen, a7=config.a7)
    return cache[key]


def _anomaly_of(*, row: dict) -> pl.DataFrame:
    """Return a series' solar anomaly table, kept for the worker's later jobs."""
    cache: dict[str, pl.DataFrame] = _CACHE["anomaly"]  # ty: ignore[invalid-assignment]
    if row["site"] not in cache:
        cache[row["site"]] = pl.read_parquet(offsets.ANOMALY_FILE).filter(
            pl.col("time_series_id") == row["time_series_id"]
        )
    return cache[row["site"]]


def train_inputs(
    *, known: pl.DataFrame, row: dict, refit: datetime, config: Config, offset: bool
) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows, weights anchored at `refit`.

    With `offset`, the base margin is the median of the training labels plus the solar offset on
    the held capacity (experiment 00064's, fitted before 2025-07-01).
    """
    age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
    label = (known["power"] / row["effective_capacity_mw"]).to_numpy()
    base_margin = None
    if offset:
        site = offsets.site_offset(site=row["site"])
        shift = offsets.offset_shift(
            offset=site,
            frame=known,
            basis="held",
            anomaly=offsets.attach_anomaly(frame=known, table=_anomaly_of(row=row)),
        )
        base_margin = offsets.margin(level=float(np.median(label)), shift=shift)
    return xgb.DMatrix(
        known.select(input_columns(a7=config.a7)).to_numpy().astype(np.float32),
        label=label,
        weight=0.5 ** (age_days.to_numpy() / config.half_life_days),
        base_margin=base_margin,
    )


def predict_inputs(
    *,
    prepared: pl.DataFrame,
    row: dict,
    level: float | None,
    basis: str,
    config: Config,
) -> xgb.DMatrix:
    """Return the prediction matrix of one run's member rows, with the offset base margin if any.

    Args:
        prepared: The run's rows for one series, sorted by valid time and member, with the lags.
        row: The series' row of `common.series_table()`.
        level: The median of the refit's training labels, or `None` for an arm with no offset.
        basis: `vintage` or `held`: the capacity of the offset's scale. Unused without a level.
        config: The configuration.

    Returns:
        The matrix, with the base margin when `level` is given.
    """
    matrix = prepared.select(input_columns(a7=config.a7)).to_numpy().astype(np.float32)
    if level is None:
        return xgb.DMatrix(matrix)
    site = offsets.site_offset(site=row["site"])
    shift = offsets.offset_shift(
        offset=site,
        frame=prepared,
        basis=basis,
        anomaly=offsets.attach_anomaly(frame=prepared, table=_anomaly_of(row=row)),
    )
    return xgb.DMatrix(matrix, base_margin=offsets.margin(level=level, shift=shift))


def prepared_run(
    *, run_rows: pl.DataFrame, row: dict, arrays: common.SeriesArrays, a7: bool
) -> pl.DataFrame:
    """Return one run's member rows for one series with derived columns and lags, sorted.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=arrays.power, a7=a7).sort(
        "valid_time", "ensemble_member"
    )
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    return prepared


def predict_variants(
    *,
    model: xgb.Booster,
    prepared: pl.DataFrame,
    row: dict,
    level: float | None,
    config: Config,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Return one run's mixture quantiles for each prediction variant of an arm.

    Args:
        model: The series' fitted booster.
        prepared: `prepared_run()`.
        row: The series' row of `common.series_table()`.
        level: The training labels' median for an offset arm, else `None`.
        config: The configuration.
        seed: The fitting seed.

    Returns:
        `{"": quantiles}` for an arm with no offset, and `{"_o": vintage, "_oheld": held}` for an
        offset arm. The second prediction changes only the capacity in the base margin.
    """
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    members = prepared["ensemble_member"].n_unique()
    variants = {"": "vintage"} if level is None else {"_o": "vintage", "_oheld": "held"}
    out = {}
    for suffix, basis in variants.items():
        matrix = predict_inputs(prepared=prepared, row=row, level=level, basis=basis, config=config)
        out[suffix] = mixture_frame(
            keys=keys, raw=model.predict(matrix), members=members, seed=seed
        )
    return out


def run_job(
    *,
    config_name: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    offset: bool,
    out_root: Path,
    rounds: int,
    nthread: int,
    seed: int,
    workers: int,
    log_hours: bool,
) -> list[str]:
    """Fit one refit for each series that lacks its file, forecast its runs, and write the files.

    Args:
        config_name: A key of `CONFIGS` (not the pooled one).
        refit: The refit time.
        origins: The `power_fcst_init_time` of the runs the refit forecasts.
        sites: The series labels to fit.
        offset: Fit the offset twin (the series must be in `offsets.OFFSET_SITES`).
        out_root: The folder holding one subfolder per configuration.
        rounds: Boosting rounds.
        nthread: Threads for each fit.
        seed: The fitting seed.
        workers: Worker processes sharing the GPU, recorded in the GPU-hour log.
        log_hours: Append the job to the GPU-hour log (false for the smoke test).

    Returns:
        One progress line per series fitted.
    """
    config = CONFIGS[config_name]
    series, arrays = _series_and_arrays()
    suffixes = ("_o", "_oheld") if offset else ("",)
    todo = [
        site
        for site in sites
        if not all(
            job_path(
                root=out_root, config=config.name + suffix, site=site, seed=seed, refit=refit
            ).exists()
            for suffix in suffixes
        )
    ]
    if not todo:
        return [f"{config.name} refit {refit:%Y-%m-%d %H:%M}: all files exist"]
    started_job = time.monotonic()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    models: dict[str, xgb.Booster] = {}
    levels: dict[str, float | None] = {}
    lines = []
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = known_rows(
            rows=_training_rows_cached(row=row, arrays=arrays[site], config=config),
            refit=refit,
            config=config,
        )
        levels[site] = (
            float(np.median((known["power"] / row["effective_capacity_mw"]).to_numpy()))
            if offset
            else None
        )
        models[site] = xgb.train(
            parameters(config=config, seed=seed, nthread=nthread),
            train_inputs(known=known, row=row, refit=refit, config=config, offset=offset),
            num_boost_round=rounds,
        )
        lines.append(
            f"{config.name}{'_o' if offset else ''} {site} refit {refit:%Y-%m-%d %H:%M}: "
            f"{known.height:,} rows, {len(origins)} runs, fit {time.monotonic() - started:.0f} s"
        )
    pieces: dict[tuple[str, str], list[pl.DataFrame]] = {}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site in todo:
            row = rows_of[site]
            prepared = prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=config.a7,
            )
            predicted = predict_variants(
                model=models[site],
                prepared=prepared,
                row=row,
                level=levels[site],
                config=config,
                seed=seed,
            )
            for suffix, table in predicted.items():
                pieces.setdefault((site, suffix), []).append(table)
    for (site, suffix), tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=job_path(
                root=out_root, config=config.name + suffix, site=site, seed=seed, refit=refit
            ),
        )
    if log_hours:
        common.log_gpu_seconds(
            job=f"{config.name}{'_o' if offset else ''} {refit:%Y-%m-%d %H:%M} {','.join(todo)}"
            f" s{seed}",
            seconds=time.monotonic() - started_job,
            workers=workers,
        )
    return lines


def pool_matrix(
    *,
    rows_of: dict[str, dict],
    arrays: dict[str, common.SeriesArrays],
    config: Config,
    refit: datetime,
    codes: dict[str, int],
) -> tuple[xgb.DMatrix, int]:
    """Return the pooled training matrix of a refit, and its width including the series code.

    Args:
        rows_of: The pool's series rows of `common.series_table()`, by site label.
        arrays: Each site's arrays on the dense grid.
        config: The pooled configuration.
        refit: The refit time.
        codes: Each site label's categorical code.

    Returns:
        The weighted matrix, whose last column is the categorical series code, and its width.
    """
    parts = []
    for site, row in rows_of.items():
        known = known_rows(
            rows=_training_rows_cached(row=row, arrays=arrays[site], config=config),
            refit=refit,
            config=config,
        )
        age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
        parts.append(
            (
                np.hstack(
                    [
                        known.select(input_columns(a7=False)).to_numpy().astype(np.float32),
                        np.full((known.height, 1), codes[site], np.float32),
                    ]
                ),
                (known["power"] / row["effective_capacity_mw"]).to_numpy(),
                0.5 ** (age_days.to_numpy() / config.half_life_days),
            )
        )
    width = parts[0][0].shape[1]
    matrix = xgb.DMatrix(
        np.vstack([p[0] for p in parts]),
        label=np.concatenate([p[1] for p in parts]),
        weight=np.concatenate([p[2] for p in parts]),
        feature_types=["q"] * (width - 1) + ["c"],
        enable_categorical=True,
    )
    return matrix, width


def run_pooled_job(
    *,
    refit: datetime,
    origins: list[datetime],
    units: str,
    out_root: Path,
    rounds: int,
    nthread: int,
    seed: int,
    workers: int,
    log_hours: bool,
    limit_sites: int | None,
) -> list[str]:
    """Fit one pooled model of the Primary series with one metered unit, one file per series.

    The series identifier is a categorical input column, coded by the series' position in
    `common.series_table()`. All of the pool's series train in one booster.

    Args:
        refit: The refit time.
        origins: The `power_fcst_init_time` of the runs the refit forecasts.
        units: `MW` or `MVA`.
        out_root: The folder holding one subfolder per configuration.
        rounds: Boosting rounds.
        nthread: Threads for the fit.
        seed: The fitting seed.
        workers: Worker processes sharing the GPU, recorded in the GPU-hour log.
        log_hours: Append the job to the GPU-hour log (false for the smoke test).
        limit_sites: Keep only this many series of the pool (the smoke test), or `None`.

    Returns:
        One progress line.
    """
    config = CONFIGS["pt"]
    series, arrays = _series_and_arrays()
    codes = {site: i for i, site in enumerate(series["site"].to_list())}
    pool = series.filter((pl.col("substation_type") == "Primary") & (pl.col("units") == units))
    if limit_sites is not None:
        pool = pool.head(limit_sites)
    rows_of = {row["site"]: row for row in pool.iter_rows(named=True)}
    targets = {
        site: job_path(root=out_root, config=config.name, site=site, seed=seed, refit=refit)
        for site in rows_of
    }
    if all(path.exists() for path in targets.values()):
        return [f"pt {units} refit {refit:%Y-%m-%d %H:%M}: all files exist"]
    started = time.monotonic()
    matrix, width = pool_matrix(
        rows_of=rows_of, arrays=arrays, config=config, refit=refit, codes=codes
    )
    rows_total = matrix.num_row()
    fit_started = time.monotonic()
    model = xgb.train(
        parameters(config=config, seed=seed, nthread=nthread), matrix, num_boost_round=rounds
    )
    fit_seconds = time.monotonic() - fit_started
    pieces: dict[str, list[pl.DataFrame]] = {site: [] for site in rows_of}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site, row in rows_of.items():
            prepared = prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=False,
            )
            members = prepared["ensemble_member"].n_unique()
            matrix = np.hstack(
                [
                    prepared.select(input_columns(a7=False)).to_numpy().astype(np.float32),
                    np.full((prepared.height, 1), codes[site], np.float32),
                ]
            )
            raw = model.predict(
                xgb.DMatrix(
                    matrix,
                    feature_types=["q"] * (width - 1) + ["c"],
                    enable_categorical=True,
                )
            )
            pieces[site].append(
                mixture_frame(
                    keys=prepared.select(*common.KEYS).unique(maintain_order=True),
                    raw=raw,
                    members=members,
                    seed=seed,
                )
            )
    for site, tables in pieces.items():
        common.atomic_write(frame=pl.concat(tables), target=targets[site])
    if log_hours:
        common.log_gpu_seconds(
            job=f"pt {units} {refit:%Y-%m-%d %H:%M} s{seed}",
            seconds=time.monotonic() - started,
            workers=workers,
        )
    line = (
        f"pt {units} refit {refit:%Y-%m-%d %H:%M}: {len(rows_of)} series, {rows_total:,} rows, "
        f"fit {fit_seconds:.0f} s, job {time.monotonic() - started:.0f} s"
    )
    return [line]


def selected_jobs(
    *, config: Config, regime: str | None, origins: list[datetime]
) -> dict[datetime, list[datetime]]:
    """Return each refit to run and the runs it forecasts.

    A refit forecasts the union of the runs that any regime of the configuration assigns to it, so
    that the file of the refit shared by the weekly and monthly regimes holds both regimes' runs
    whichever regime is fitted first.

    Args:
        config: The configuration.
        regime: Fit only this regime's refits, or `None` for every regime of the configuration.
        origins: The `power_fcst_init_time` of the runs to cover.

    Returns:
        Refit times in ascending order mapped to their runs.
    """
    union = regimes.jobs_for(regime_names=config.regime_names, origins=origins)
    if regime is None:
        return union
    wanted = set(regimes.REGIMES[regime])
    return {refit: runs for refit, runs in union.items() if refit in wanted}


def _dry_pooled(*, config: Config, nthread: int, seed: int) -> None:
    """Time the MW pool's fit at the first and last refit and project both pools' GPU hours."""
    series, arrays = _series_and_arrays()
    codes = {site: i for i, site in enumerate(series["site"].to_list())}
    refits = sorted(selected_jobs(config=config, regime="monthly", origins=common.all_origins()))
    for units in ("MW", "MVA"):
        pool = series.filter((pl.col("substation_type") == "Primary") & (pl.col("units") == units))
        rows_of = {row["site"]: row for row in pool.iter_rows(named=True)}
        seconds = []
        for refit in (refits[0], refits[-1]):
            matrix, _ = pool_matrix(
                rows_of=rows_of, arrays=arrays, config=config, refit=refit, codes=codes
            )
            started = time.monotonic()
            xgb.train(
                parameters(config=config, seed=seed, nthread=nthread),
                matrix,
                num_boost_round=config.rounds,
            )
            seconds.append(time.monotonic() - started)
            print(
                f"  pt {units} ({len(rows_of)} series) {refit:%Y-%m-%d}: "
                f"{matrix.num_row():,} rows, {seconds[-1]:.0f} s"
            )
        mean = float(np.mean(seconds))
        print(
            f"  pt {units}: {len(refits)} refits x {mean:.0f} s = "
            f"{len(refits) * mean / 3600.0:.2f} GPU-h"
        )


def dry_gpu_hours(*, config: Config, nthread: int, seed: int) -> None:
    """Time two real fits of a configuration and project the GPU hours of its full run.

    The fits are one series at the first and one at the last refit of the configuration, with the
    configuration's full rounds. The projection multiplies the mean of the two timings by the
    number of fits of each group. It overstates the busy time of a shared GPU when workers run
    side by side, because it counts each fit in full, as the GPU-hour log does.
    """
    series, arrays = _series_and_arrays()
    origins = common.all_origins()
    print(f"{config.name}: timing real fits with {config.rounds} rounds, {nthread} threads")
    if config.pooled:
        _dry_pooled(config=config, nthread=nthread, seed=seed)
        return
    for regime in config.regime_names:
        refits = sorted(selected_jobs(config=config, regime=regime, origins=origins))
        probe_site = series.filter(pl.col("substation_type") == "Primary")["site"][0]
        row = series.filter(pl.col("site") == probe_site).row(0, named=True)
        seconds = []
        for refit in (refits[0], refits[-1]):
            known = known_rows(
                rows=_training_rows_cached(row=row, arrays=arrays[probe_site], config=config),
                refit=refit,
                config=config,
            )
            started = time.monotonic()
            xgb.train(
                parameters(config=config, seed=seed, nthread=nthread),
                train_inputs(known=known, row=row, refit=refit, config=config, offset=False),
                num_boost_round=config.rounds,
            )
            seconds.append(time.monotonic() - started)
            print(
                f"  {regime} {probe_site} {refit:%Y-%m-%d}: {known.height:,} rows, "
                f"{seconds[-1]:.0f} s"
            )
        mean = float(np.mean(seconds))
        for group in config.groups:
            count = series.filter(pl.col("substation_type") == group).height
            hours = len(refits) * count * mean / 3600.0
            print(
                f"  {regime} {group}: {len(refits)} refits x {count} series x {mean:.0f} s "
                f"= {hours:.2f} GPU-h (fits only, no offset twin)"
            )


def main() -> None:
    """Fit and write the requested configuration's jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--regime", choices=REGIME_NAMES, help="fit only this regime's refits")
    parser.add_argument("--groups", nargs="+", choices=common.GROUPS)
    parser.add_argument("--sites", nargs="+", help="fit only these series labels, e.g. D2 D5")
    parser.add_argument("--offset", action="store_true", help="the offset twin at the 5 Primaries")
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--workers", type=int, default=2, help="processes sharing the GPU")
    parser.add_argument("--cores", type=int, default=12, help="CPU cores for all workers")
    parser.add_argument(
        "--smoke", action="store_true", help="one series, 2 refits, 3 runs, 20 rounds, scratch dir"
    )
    parser.add_argument(
        "--dry-gpu-hours", action="store_true", help="time two real fits and project the GPU hours"
    )
    args = parser.parse_args()
    config = CONFIGS[args.config]
    nthread = max(1, args.cores // args.workers)
    if args.dry_gpu_hours:
        dry_gpu_hours(config=config, nthread=nthread, seed=args.seed)
        return
    print(f"GPU hours logged so far: {common.gpu_hours_total():.2f} of 16", flush=True)
    groups = tuple(args.groups) if args.groups else config.groups
    series = common.series_table().filter(pl.col("substation_type").is_in(groups))
    if args.offset:
        if config.pooled:
            msg = "the pooled configuration has no offset twin"
            raise SystemExit(msg)
        series = series.filter(pl.col("site").is_in(list(offsets.OFFSET_SITES)))
    if args.sites:
        series = series.filter(pl.col("site").is_in(args.sites))
    sites = series["site"].to_list()
    jobs = selected_jobs(config=config, regime=args.regime, origins=common.all_origins())
    out_root = common.ANCHOR_PREDS_DIR
    rounds, workers = config.rounds, args.workers
    if args.smoke:
        sites = sites[:1]
        jobs = {refit: runs[:SMOKE_RUNS] for refit, runs in list(jobs.items())[:SMOKE_REFITS]}
        out_root = out_root.with_name("anchor_preds_smoke")
        rounds, workers = SMOKE_ROUNDS, 1
        nthread = args.cores
    started = time.monotonic()
    ordered = sorted(jobs.items(), key=lambda item: -len(item[1]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if config.pooled:
            futures = [
                pool.submit(
                    run_pooled_job,
                    refit=refit,
                    origins=runs,
                    units=units,
                    out_root=out_root,
                    rounds=rounds,
                    nthread=nthread,
                    seed=args.seed,
                    workers=workers,
                    log_hours=not args.smoke,
                    limit_sites=2 if args.smoke else None,
                )
                for refit, runs in ordered
                for units in ("MW", "MVA")
            ]
        else:
            # Fewer refits than workers (the frozen model has one): split the series so every
            # worker works.
            n_chunks = workers if len(ordered) < workers else 1
            site_chunks = [sites[i::n_chunks] for i in range(n_chunks)]
            futures = [
                pool.submit(
                    run_job,
                    config_name=config.name,
                    refit=refit,
                    origins=runs,
                    sites=chunk,
                    offset=args.offset,
                    out_root=out_root,
                    rounds=rounds,
                    nthread=nthread,
                    seed=args.seed,
                    workers=workers,
                    log_hours=not args.smoke,
                )
                for refit, runs in ordered
                for chunk in site_chunks
                if chunk
            ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(
        f"done in {time.monotonic() - started:.0f} s; "
        f"GPU hours logged: {common.gpu_hours_total():.2f} of 16"
    )


if __name__ == "__main__":
    main()
