"""Fit the retraining-cadence arms of experiment 00034 and write each refit's validation quantiles.

One job is one refit of one series for one seed and one configuration. A refit at time `t` trains
on the ENS control member's rows whose valid time is at least 3.5 hours before `t` and whose run
began at or before `t`, weighted by a recency half-life anchored at `t`, and then forecasts the
validation runs that any regime assigns to it (`e00034_regimes`). The job writes the mixture of the
51 members' nine quantiles for those runs to `work/anchor_preds/<config>/`, keyed by series, run,
and valid time. Existing files stay, so a stopped run resumes.

Configurations:

- `n1h`: experiment 33's monthly recipe (half-life 0.5 years, 500 rounds, `eta` 0.05).
- `n1_2y`: the same with a half-life of 2 years.
- `roll12`: the rolling-window proxy (latest 12 months of rows, 150 rounds, `eta` 0.15).

The first monthly refit of `n1h` is also saved as a booster, the starting point of the warm-start
proxy (`e00034_fit_warm.py`).

    uv run python ml-experiments/00034-*/e00034_fit.py --config n1h --regimes monthly \
        --groups Primary --seeds 0 1 2 [--smoke]
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00034_common as common
import e00034_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

ANCHOR_PREDS_DIR: Final[Path] = common.WORK_DIR / "anchor_preds"
BOOSTERS_DIR: Final[Path] = common.WORK_DIR / "boosters"
AFTER_DIR: Final[Path] = common.WORK_DIR / "ens_after_training"
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)


@dataclass(frozen=True)
class Config:
    """A training configuration.

    Attributes:
        name: The folder name under `anchor_preds`.
        half_life_days: Recency half-life of the row weights, anchored at the refit time.
        window_days: Keep only rows whose valid time is within this many days of the refit, or
            `None` for the whole history.
        rounds: Boosting rounds.
        eta: Learning rate.
    """

    name: str
    half_life_days: float
    window_days: float | None
    rounds: int
    eta: float


CONFIGS: Final[dict[str, Config]] = {
    "n1h": Config("n1h", common.SHORT_HALF_LIFE_DAYS, None, common.N_ROUNDS, 0.05),
    "n1_2y": Config("n1_2y", common.RECENCY_HALF_LIFE_DAYS, None, common.N_ROUNDS, 0.05),
    "roll12": Config("roll12", common.SHORT_HALF_LIFE_DAYS, 365.0, 150, 0.15),
}


def input_columns() -> list[str]:
    """Return the input columns in matrix order."""
    return [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags to a frame with a run's origin and valid times."""
    lags = common.origin_lags(
        power=power,
        target=np.asarray(common.grid_index(times=frame["valid_time"])),
        origin=np.asarray(common.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's nine quantiles, as experiment 00019 does."""
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(common.ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def training_rows(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return one series' control-member rows to 2026-06-29 with derived columns and lags.

    The power of the rows after 2025-06-30 is filled from the dense grid. Every row's refit
    eligibility (valid time and run start) is applied per refit, not here.
    """
    series_id = row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *WEATHER)
        .collect()
    )
    after = pl.concat(
        [
            pl.read_parquet(path).filter(pl.col("time_series_id") == series_id)
            for path in sorted(AFTER_DIR.glob("init=*.parquet"))
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
    return add_lags(frame=rows, power=arrays.power)


def known_rows(*, rows: pl.DataFrame, refit: datetime, window_days: float | None) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see."""
    known = rows.filter(
        (pl.col("valid_time") <= refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("power_fcst_init_time") <= refit)
    )
    if window_days is not None:
        known = known.filter(pl.col("valid_time") > refit - timedelta(days=window_days))
    return known


def train_matrix(
    *, rows: pl.DataFrame, row: dict, refit: datetime, half_life_days: float
) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows, weights anchored at `refit`."""
    age_days = (refit - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns()).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / half_life_days),
    )


def fit(*, matrix: xgb.DMatrix, seed: int, config: Config) -> xgb.Booster:
    """Fit one quantile model."""
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": config.eta},
        matrix,
        num_boost_round=config.rounds,
    )


def prepared_runs(
    *, row: dict, arrays: common.SeriesArrays, origins: set[datetime]
) -> dict[datetime, tuple[pl.DataFrame, np.ndarray, int]]:
    """Return each needed validation run's keys, input matrix, and member count for one series."""
    out = {}
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        origin = run["power_fcst_init_time"][0]
        if origin not in origins:
            continue
        run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
        run = add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        if run.height != run.select(*common.KEYS).n_unique() * members:
            msg = "every valid time must hold every member"
            raise ValueError(msg)
        out[origin] = (
            run.select(*common.KEYS).unique(maintain_order=True),
            run.select(input_columns()).to_numpy().astype(np.float32),
            members,
        )
    return out


def predict_runs(
    *,
    model: xgb.Booster,
    prepared: dict[datetime, tuple[pl.DataFrame, np.ndarray, int]],
    origins: list[datetime],
) -> pl.DataFrame:
    """Return the mixture quantiles of the named validation runs."""
    pieces = []
    for origin in origins:
        keys, matrix, members = prepared[origin]
        raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, members, len(common.LEVELS))
        pieces.append(
            keys.hstack(
                pl.DataFrame(
                    mixture_quantiles(member_quantiles=raw), schema=list(common.LEVEL_COLUMNS)
                )
            )
        )
    return pl.concat(pieces)


def job_path(*, root: Path, config: str, site: str, seed: int, refit: datetime) -> Path:
    """Return the file one refit's predictions are written to."""
    return root / config / f"{site}_s{seed}_{refit:%Y%m%dT%H%M}.parquet"


def build_jobs(*, regime_group: str, origins: list[datetime]) -> dict[datetime, list[datetime]]:
    """Return each refit time and the origins to predict, for one group of regimes.

    The `monthly` group is the monthly, half-yearly, three-a-year, quarterly, and yearly regimes.
    `weekly` and `daily` are groups of their own and are written to their own folders, because the
    weekly and monthly regimes share the refit 2025-07-01 with different sets of origins.
    """
    if regime_group == "monthly":
        jobs = regimes.monthly_family_jobs(origins=origins)
    else:
        jobs = regimes.served(name=regime_group, origins=origins)
    return {refit: sorted(o) for refit, o in sorted(jobs.items())}


def folder_for(*, config: str, regime_group: str) -> str:
    """Return the folder under `anchor_preds` for a configuration and a group of regimes."""
    return config if regime_group == "monthly" else f"{config}_{regime_group}"


def main() -> None:
    """Fit and write the requested jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=("monthly", "weekly", "daily"),
        required=True,
        help="`monthly` covers the monthly, half-yearly, three-a-year, quarterly, yearly regimes",
    )
    parser.add_argument("--groups", nargs="+", default=["Primary"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--smoke", action="store_true", help="two series, a few jobs, scratch dir")
    args = parser.parse_args()
    config = CONFIGS[args.config]
    series = common.series_table().filter(pl.col("substation_type").is_in(args.groups))
    if args.smoke:
        series = series.head(2)
    origins = regimes.run_origins()
    arrays = common.load_series_arrays(series=series)
    out_root = ANCHOR_PREDS_DIR.with_name("anchor_preds_smoke") if args.smoke else ANCHOR_PREDS_DIR
    all_jobs = {}
    for group in args.regimes:
        jobs = build_jobs(regime_group=group, origins=origins)
        if args.smoke:
            jobs = dict(list(jobs.items())[:2])
        all_jobs[group] = jobs
    needed = {o for jobs in all_jobs.values() for served in jobs.values() for o in served}
    for row in series.iter_rows(named=True):
        site = arrays[row["site"]]
        todo = [
            (group, seed, refit)
            for group, jobs in all_jobs.items()
            for seed in args.seeds
            for refit in jobs
            if not job_path(
                root=out_root,
                config=folder_for(config=config.name, regime_group=group),
                site=row["site"],
                seed=seed,
                refit=refit,
            ).exists()
        ]
        if not todo:
            continue
        rows = training_rows(row=row, arrays=site)
        prepared = prepared_runs(row=row, arrays=site, origins=needed)
        for group, seed, refit in todo:
            started = time.monotonic()
            known = known_rows(rows=rows, refit=refit, window_days=config.window_days)
            model = fit(
                matrix=train_matrix(
                    rows=known, row=row, refit=refit, half_life_days=config.half_life_days
                ),
                seed=seed,
                config=config,
            )
            if config.name == "n1h" and refit == regimes.MONTHLY[0]:
                folder = BOOSTERS_DIR.with_name("boosters_smoke") if args.smoke else BOOSTERS_DIR
                folder.mkdir(parents=True, exist_ok=True)
                model.save_model(folder / f"{row['site']}_s{seed}_frozen.ubj")
            frame = predict_runs(model=model, prepared=prepared, origins=all_jobs[group][refit])
            frame = frame.with_columns(seed=pl.lit(seed).cast(pl.Int8))
            target = job_path(
                root=out_root,
                config=folder_for(config=config.name, regime_group=group),
                site=row["site"],
                seed=seed,
                refit=refit,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
            print(
                f"{config.name} {group} {row['site']} seed {seed} refit {refit:%Y-%m-%d %H:%M}: "
                f"{known.height:,} rows, {len(all_jobs[group][refit])} runs, "
                f"{time.monotonic() - started:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
