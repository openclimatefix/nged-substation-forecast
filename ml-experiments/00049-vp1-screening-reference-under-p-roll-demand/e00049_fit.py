"""Fit the stage 1 arms of experiment 00049 under P-roll; write each refit's quantiles.

P-roll refits monthly (2025-07-01 to 2026-06-01) on the latest 12 months of the ENS control
member's rows, with 150 rounds and `eta` 0.15, and forecasts the validation runs that the monthly
regime assigns to the refit. One job is one refit of one series (or, for the pooled arms, of one
pool) for seed 0. The job writes the mixture of the 51 members' nine quantiles to
`work/anchor_preds/<arm>/`, keyed by series, run, and valid time. Existing files stay, so a stopped
run resumes.

Per-series arms (one XGBoost model per series): `c0` (the reference), `no_christmas`, `no_covid`,
`hl2y` (half-life of 2 years in place of 0.5).

Pooled arms (one XGBoost model per metered-units pool, series identifier as a categorical column):
`c0b` (the reference) and `f1` (plus experiment 00032's training-window and recent-window
statistics). A refit's training-window statistics use the 365 days of actuals ending at the refit's
cutoff (3.5 hours before the refit); a case's recent-window statistics use the 14 days of actuals
ending 2 days before its run start.

    uv run python ml-experiments/00049-*/e00049_fit.py --arms c0 no_christmas --shard 0 2 [--smoke]
    uv run python ml-experiments/00049-*/e00049_fit.py --arms c0b f1 --pools MW [--smoke]
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00049_common as common
import e00049_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb

ANCHOR_PREDS_DIR: Path = common.WORK_DIR / "anchor_preds"
AFTER_DIR: Final[Path] = common.WORK_DIR_00034 / "ens_after_training"
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
WINDOW_DAYS: Final[float] = 365.0
ROUNDS: Final[int] = 150
ETA: Final[float] = 0.15
THREADS: Final[int] = 1
"""XGBoost threads per worker; the slot has 2 cores and runs 2 workers."""
SEED: Final[int] = 0
STEPS_PER_DAY: Final[int] = 48
GRID_START_WEEKDAY: Final[int] = 6
"""2019-09-01, the grid start, is a Sunday (Monday is 0)."""
RECENT_DAYS: Final[int] = 14
RECENT_GAP: Final[int] = 2 * STEPS_PER_DAY
RECENT_MIN_COUNT: Final[int] = 3 * STEPS_PER_DAY
DAY_SLOTS: Final[slice] = slice(20, 32)
NIGHT_SLOTS: Final[slice] = slice(0, 10)
"""Half-hour-of-day slots (period-ending UTC stamps): 10:00 to 15:30 and 00:00 to 04:30."""


@dataclass(frozen=True)
class Arm:
    """One stage 1 arm.

    Attributes:
        name: The folder name under `anchor_preds`.
        half_life_days: Recency half-life of the row weights, anchored at the refit time.
        drop: Input columns removed from the frozen set.
        pooled: Fit one model per metered-units pool, with the series identifier.
        fingerprints: Add experiment 00032's training-window and recent-window statistics.
    """

    name: str
    half_life_days: float = common.SHORT_HALF_LIFE_DAYS
    drop: tuple[str, ...] = ()
    pooled: bool = False
    fingerprints: bool = False


ARMS: Final[dict[str, Arm]] = {
    "c0": Arm("c0"),
    "no_christmas": Arm("no_christmas", drop=("days_to_christmas",)),
    "no_covid": Arm("no_covid", drop=("covid_indicator",)),
    "hl2y": Arm("hl2y", half_life_days=common.RECENCY_HALF_LIFE_DAYS),
    "c0b": Arm("c0b", pooled=True),
    "f1": Arm("f1", pooled=True, fingerprints=True),
}
TRAIN_STAT_COUNT: Final[int] = 7
RECENT_STAT_COUNT: Final[int] = 3


def input_columns(*, drop: tuple[str, ...] = ()) -> list[str]:
    """Return the input columns in matrix order, without the dropped ones."""
    return [
        *(c for c in common.feature_columns(calendar=True) if c not in drop),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags to a frame of runs."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    return frame.with_columns(
        pl.DataFrame(
            common.origin_lags(power=power, target=target, origin=origin),
            schema=list(common.LAG_COLUMNS),
            orient="row",
        ).cast(pl.Float32)
    )


def training_fingerprint(*, power: np.ndarray, refit: datetime) -> np.ndarray:
    """Return the 7 training-window statistics of one series for a refit.

    Uses the observations in the 365 days ending at the refit's cutoff, 3.5 hours before the refit.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        refit: The refit time.

    Returns:
        A float32 vector: mean, standard deviation, 5th and 95th percentiles, day-minus-night
        mean, weekend-minus-weekday mean, and the half-hour slot of the profile's peak.
    """
    stop = int(common.grid_index(times=refit - regimes.REFIT_CUTOFF_AGE)) + 1
    start = stop - int(WINDOW_DAYS) * STEPS_PER_DAY
    values = power[start:stop].astype(np.float64)
    index = np.arange(start, stop)
    slot = index % STEPS_PER_DAY
    weekend = (GRID_START_WEEKDAY + index // STEPS_PER_DAY) % 7 >= 5
    good = np.isfinite(values)
    day = good & (slot >= DAY_SLOTS.start) & (slot < DAY_SLOTS.stop)
    night = good & (slot >= NIGHT_SLOTS.start) & (slot < NIGHT_SLOTS.stop)
    profile = np.array(
        [
            values[good & (slot == k)].mean() if (good & (slot == k)).any() else np.nan
            for k in range(STEPS_PER_DAY)
        ]
    )
    return np.array(
        [
            values[good].mean(),
            values[good].std(),
            np.percentile(values[good], 5),
            np.percentile(values[good], 95),
            values[day].mean() - values[night].mean(),
            values[good & weekend].mean() - values[good & ~weekend].mean(),
            float(np.nanargmax(profile)),
        ],
        dtype=np.float32,
    )


def recent_fingerprint(*, power: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the 3 recent-window statistics before each forecast origin.

    The window covers the 14 days ending 2 days before the origin.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        origin: Grid index of each case's run start.

    Returns:
        A float32 array of shape (cases, 3): mean, standard deviation, and day-minus-night mean,
        NaN where the window holds fewer than 3 days of observations.
    """
    values = np.nan_to_num(power.astype(np.float64))
    good = np.isfinite(power).astype(np.float64)
    slot = np.arange(power.size) % STEPS_PER_DAY
    day = ((slot >= DAY_SLOTS.start) & (slot < DAY_SLOTS.stop)).astype(np.float64)
    night = ((slot >= NIGHT_SLOTS.start) & (slot < NIGHT_SLOTS.stop)).astype(np.float64)

    def prefix(x: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(x)])

    total, count, squares = prefix(values), prefix(good), prefix(values**2)
    day_total, day_count = prefix(values * day), prefix(good * day)
    night_total, night_count = prefix(values * night), prefix(good * night)
    high = np.clip(origin - RECENT_GAP, 0, power.size)
    low = np.clip(high - RECENT_DAYS * STEPS_PER_DAY, 0, power.size)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[high] - total[low]) / n
        variance = (squares[high] - squares[low]) / n - mean**2
        day_mean = (day_total[high] - day_total[low]) / (day_count[high] - day_count[low])
        night_mean = (night_total[high] - night_total[low]) / (night_count[high] - night_count[low])
    out = np.stack([mean, np.sqrt(np.maximum(variance, 0.0)), day_mean - night_mean], axis=1)
    out[n < RECENT_MIN_COUNT] = np.nan
    return out.astype(np.float32)


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


def known_rows(*, rows: pl.DataFrame, refit: datetime) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see: the latest 12 months, eligible at the refit."""
    return rows.filter(
        (pl.col("valid_time") <= refit - regimes.REFIT_CUTOFF_AGE)
        & (pl.col("power_fcst_init_time") <= refit)
        & (pl.col("valid_time") > refit - timedelta(days=WINDOW_DAYS))
    )


def extra_columns(
    *, arm: Arm, frame: pl.DataFrame, power: np.ndarray, refit: datetime
) -> np.ndarray:
    """Return the fingerprint columns of a frame of cases, or an empty width for other arms."""
    if not arm.fingerprints:
        return np.empty((frame.height, 0), np.float32)
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    training = training_fingerprint(power=power, refit=refit)
    return np.hstack(
        [np.tile(training, (frame.height, 1)), recent_fingerprint(power=power, origin=origin)]
    )


def prepared_runs(
    *, row: dict, arrays: common.SeriesArrays, origins: set[datetime], drop: tuple[str, ...]
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
            run.select(input_columns(drop=drop)).to_numpy().astype(np.float32),
            members,
        )
    return out


def parameters() -> dict[str, object]:
    """Return N1's XGBoost parameters with P-roll's learning rate and this slot's thread count."""
    return {**common.QUANTILE_PARAMETERS, "seed": SEED, "eta": ETA, "nthread": THREADS}


def job_path(*, arm: str, name: str, refit: datetime) -> Path:
    """Return the file one refit's predictions are written to (`name` is a site or a pool)."""
    return ANCHOR_PREDS_DIR / arm / f"{name}_s{SEED}_{refit:%Y%m%dT%H%M}.parquet"


def write_atomic(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file through a temporary name so a crash leaves no partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)


def predictions(
    *,
    model: xgb.Booster,
    keys: pl.DataFrame,
    matrix: np.ndarray,
    members: int,
) -> pl.DataFrame:
    """Return one run's mixture quantiles with its keys and the seed."""
    raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, members, len(common.LEVELS))
    return keys.hstack(
        pl.DataFrame(mixture_quantiles(member_quantiles=raw), schema=list(common.LEVEL_COLUMNS))
    ).with_columns(seed=pl.lit(SEED).cast(pl.Int8))


def run_per_series(*, arm: Arm, series: pl.DataFrame, jobs: dict[datetime, list[datetime]]) -> None:
    """Fit and write a per-series arm for the given series."""
    arrays = common.load_series_arrays(series=series)
    needed = {o for served in jobs.values() for o in served}
    for row in series.iter_rows(named=True):
        site = arrays[row["site"]]
        todo = [r for r in jobs if not job_path(arm=arm.name, name=row["site"], refit=r).exists()]
        if not todo:
            continue
        rows = training_rows(row=row, arrays=site)
        prepared = prepared_runs(row=row, arrays=site, origins=needed, drop=arm.drop)
        for refit in todo:
            started = time.monotonic()
            known = known_rows(rows=rows, refit=refit)
            age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
            matrix = xgb.DMatrix(
                known.select(input_columns(drop=arm.drop)).to_numpy().astype(np.float32),
                label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
                weight=0.5 ** (age_days.to_numpy() / arm.half_life_days),
            )
            model = xgb.train(parameters(), matrix, num_boost_round=ROUNDS)
            frame = pl.concat(
                [
                    predictions(model=model, keys=k, matrix=m, members=n)
                    for k, m, n in (prepared[o] for o in jobs[refit])
                ]
            )
            write_atomic(frame=frame, target=job_path(arm=arm.name, name=row["site"], refit=refit))
            print(
                f"{arm.name} {row['site']} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
                f"{time.monotonic() - started:.0f} s",
                flush=True,
            )


def run_pooled(
    *, arm: Arm, pool: pl.DataFrame, units: str, jobs: dict[datetime, list[datetime]]
) -> None:
    """Fit and write a pooled arm for one metered-units pool of Primary substations."""
    codes = {s: i for i, s in enumerate(common.series_table()["site"].to_list())}
    arrays = common.load_series_arrays(series=pool)
    needed = {o for served in jobs.values() for o in served}
    rows, prepared = {}, {}
    for row in pool.iter_rows(named=True):
        rows[row["site"]] = training_rows(row=row, arrays=arrays[row["site"]])
        prepared[row["site"]] = prepared_runs(
            row=row, arrays=arrays[row["site"]], origins=needed, drop=arm.drop
        )
    for refit, origins_of_refit in jobs.items():
        targets = {
            r["site"]: job_path(arm=arm.name, name=r["site"], refit=refit)
            for r in pool.iter_rows(named=True)
        }
        if all(p.exists() for p in targets.values()):
            continue
        started = time.monotonic()
        parts = []
        for r in pool.iter_rows(named=True):
            known = known_rows(rows=rows[r["site"]], refit=refit)
            age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
            base = known.select(input_columns(drop=arm.drop)).to_numpy().astype(np.float32)
            extra = extra_columns(arm=arm, frame=known, power=arrays[r["site"]].power, refit=refit)
            code = np.full((base.shape[0], 1), codes[r["site"]], np.float32)
            parts.append(
                (
                    np.hstack([base, extra, code]),
                    (known["power"] / r["effective_capacity_mw"]).to_numpy(),
                    0.5 ** (age_days.to_numpy() / arm.half_life_days),
                )
            )
        width = parts[0][0].shape[1]
        model = xgb.train(
            parameters(),
            xgb.DMatrix(
                np.vstack([p[0] for p in parts]),
                label=np.concatenate([p[1] for p in parts]),
                weight=np.concatenate([p[2] for p in parts]),
                feature_types=["q"] * (width - 1) + ["c"],
                enable_categorical=True,
            ),
            num_boost_round=ROUNDS,
        )
        for r in pool.iter_rows(named=True):
            pieces = []
            for origin in origins_of_refit:
                keys, matrix, members = prepared[r["site"]][origin]
                case_frame = keys.select("power_fcst_init_time")
                extra = extra_columns(
                    arm=arm, frame=case_frame, power=arrays[r["site"]].power, refit=refit
                )
                full = np.hstack(
                    [
                        matrix,
                        np.repeat(extra, members, axis=0),
                        np.full((matrix.shape[0], 1), codes[r["site"]], np.float32),
                    ]
                )
                raw = model.predict(
                    xgb.DMatrix(
                        full,
                        feature_types=["q"] * (full.shape[1] - 1) + ["c"],
                        enable_categorical=True,
                    )
                ).reshape(-1, members, len(common.LEVELS))
                pieces.append(
                    keys.hstack(
                        pl.DataFrame(
                            mixture_quantiles(member_quantiles=raw),
                            schema=list(common.LEVEL_COLUMNS),
                        )
                    ).with_columns(seed=pl.lit(SEED).cast(pl.Int8))
                )
            write_atomic(frame=pl.concat(pieces), target=targets[r["site"]])
        print(
            f"{arm.name} {units} refit {refit:%Y-%m-%d}: {time.monotonic() - started:.0f} s",
            flush=True,
        )


def main() -> None:
    """Fit and write the requested arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    parser.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("I", "N"))
    parser.add_argument("--pools", nargs="+", choices=("MW", "MVA"), default=["MW", "MVA"])
    parser.add_argument("--smoke", action="store_true", help="two series, two refits, scratch dir")
    args = parser.parse_args()
    global ANCHOR_PREDS_DIR  # noqa: PLW0603
    if args.smoke:
        ANCHOR_PREDS_DIR = common.WORK_DIR / "anchor_preds_smoke"
    primary = common.series_table().filter(pl.col("substation_type") == "Primary")
    jobs = regimes.served(name="monthly", origins=regimes.run_origins())
    jobs = {r: sorted(o) for r, o in sorted(jobs.items())}
    if args.smoke:
        jobs = dict(list(jobs.items())[:2])
    index, count = args.shard
    for name in args.arms:
        arm = ARMS[name]
        if arm.pooled:
            for units in args.pools:
                pool = primary.filter(pl.col("units") == units)
                if args.smoke:
                    pool = pool.head(2)
                run_pooled(arm=arm, pool=pool, units=units, jobs=jobs)
        else:
            shard = primary.filter(pl.col("site").is_in(primary["site"].to_list()[index::count]))
            if args.smoke:
                shard = shard.head(2)
            run_per_series(arm=arm, series=shard, jobs=jobs)


if __name__ == "__main__":
    main()
