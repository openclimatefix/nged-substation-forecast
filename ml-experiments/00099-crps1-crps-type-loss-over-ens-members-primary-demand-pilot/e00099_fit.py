"""Fit the four arms of experiment 99 and write each refit's 11 quantiles per daily run.

The four arms differ in the training rows and in the loss, and share everything else: the 16
Primary substation series, the input columns of the best guess (BG2, experiment 87), the monthly
refit times from 2025-07-01, a rolling 12-month window, 150 rounds, `eta` 0.15 (the P-roll recipe
of experiment 85), recency weights with a half-life of 0.5 years, and a starting value for each
quantile level equal to that quantile of the window's labels.

- `p1`: the built-in pinball loss on the control member's rows only (`K = 1`), the recipe of BG2's
  weekly arm at the cheaper P-roll schedule.
- `p2`: the built-in pinball loss on the control member and five drawn members (`K = 6`).
- `p3`: a pinball loss written as a custom objective with a constant Hessian, on the same rows as
  `p2`. It exists to separate the effect of the optimiser from the effect of the loss.
- `c1`: the mixture CRPS of `e00099_objective`, on the same rows as `p2` and `p3`.

A refit at time `t` trains on rows whose valid time is within 365 days before `t` and at least 3.5
hours old at `t`, and whose run began at or before `t`. Only every `RUN_STRIDE`-th daily run
(counting from 2024-04-01) contributes training rows. Each refit forecasts the daily runs whose
latest refit it is, with all 51 members, and pools the members' atoms into 11 quantiles: the
pinball arms with the masses of experiment 87 and `c1` with `CRPS_MASSES`.

Leakage checks, each raising `ValueError`: every lag is no younger than the newest usable
observation (`e00099_common.origin_lags`), and a refit's training rows satisfy the valid-time and
run-start rules (`known_rows`).

    uv run python ml-experiments/00099-*/e00099_fit.py --arms p1 p2 p3 c1 [--workers 6]
        [--sites D2 D3] [--smoke]
"""

import argparse
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00099_common as common
import e00099_objective as objective
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
MEMBERS_TRAIN_DIR: Final[Path] = common.WORK_DIR / "members_train"
MEMBERS_AFTER_DIR: Final[Path] = common.WORK_DIR / "members_after"
PREDS_DIR: Final[Path] = common.WORK_DIR / "preds"
FIRST_TRAIN_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
RUN_STRIDE: Final[int] = 3
"""Only every third daily run (counting from 2024-04-01) contributes training rows."""
WINDOW_DAYS: Final[float] = 365.0
ROUNDS: Final[int] = 150
ETA: Final[float] = 0.15
HALF_LIFE_DAYS: Final[float] = common.SHORT_HALF_LIFE_DAYS
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
HESSIAN: Final[float] = 10.0
"""The constant Hessian of the custom objectives is `HESSIAN / 2` per row and level."""
MONTHLY: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve monthly refit times, 2025-07-01 to 2026-06-01."""
SMOKE_ROUNDS: Final[int] = 10
SMOKE_REFITS: Final[int] = 1

TREE_PARAMETERS: Final[dict[str, object]] = {
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
}


@dataclass(frozen=True)
class Arm:
    """One arm.

    Attributes:
        name: The folder name under `preds`.
        members: Rows per group: 1 for the control member only, else `MEMBERS_PER_RUN`.
        loss: `pinball` (built in), `pinball_custom`, or `crps`.
        masses: The atom masses the arm's members are pooled with at prediction time.
    """

    name: str
    members: int
    loss: str
    masses: np.ndarray


ARMS: Final[dict[str, Arm]] = {
    "p1": Arm("p1", 1, "pinball", common.ATOM_MASSES),
    "p2": Arm("p2", common.MEMBERS_PER_RUN, "pinball", common.ATOM_MASSES),
    "p3": Arm("p3", common.MEMBERS_PER_RUN, "pinball_custom", common.ATOM_MASSES),
    "c1": Arm("c1", common.MEMBERS_PER_RUN, "crps", objective.CRPS_MASSES),
}
POOLED_AGAIN: Final[dict[str, tuple[str, np.ndarray]]] = {"p3m": ("p3", objective.CRPS_MASSES)}
"""Extra output sets pooled from another arm's atoms with other masses: `p3m` is arm `p3`'s model
pooled with the CRPS masses, so that the verdict contrast can be repeated with the pooling fixed."""


def input_columns() -> list[str]:
    """Return the input columns in matrix order."""
    return [*common.feature_columns(), *common.N1_EXTRA_COLUMNS, *common.LAG_COLUMNS]


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


def in_training_runs(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows of every `RUN_STRIDE`-th daily run, counting from 2024-04-01."""
    run_day = (
        pl.col("power_fcst_init_time") - pl.lit(common.origin_of(run=FIRST_TRAIN_RUN))
    ).dt.total_days()
    return frame.filter(run_day % RUN_STRIDE == 0)


def training_rows(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return one series' training rows: the drawn members of every `RUN_STRIDE`-th daily run.

    The rows run from 2024-04-01 to 2026-06-30. Every row's eligibility for a refit is applied per
    refit, not here. The label is the observed power on the dense grid, in MW.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.

    Returns:
        The rows keyed by series, run, valid time, and member, with `power`, the weather columns,
        the derived columns, `lead_hours`, and the three lags, sorted by run, valid time, member.
    """
    rows = pl.concat(
        [
            pl.scan_parquet(MEMBERS_TRAIN_DIR / "*.parquet"),
            pl.scan_parquet(MEMBERS_AFTER_DIR / "*.parquet"),
        ]
    ).filter(pl.col("time_series_id") == row["time_series_id"])
    rows = in_training_runs(frame=rows.collect())
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    power = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
    rows = rows.with_columns(power=pl.Series(power)).filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=rows, power=arrays.power).sort(
        "power_fcst_init_time", "valid_time", "ensemble_member"
    )


def known_rows(*, rows: pl.DataFrame, refit: datetime) -> pl.DataFrame:
    """Return the rows a refit may see, and check that none comes from its future.

    Raises:
        ValueError: If no row is usable, or a kept row breaks the valid-time or run-start rule.
    """
    latest_valid = refit - REFIT_CUTOFF_AGE
    known = rows.filter(
        (pl.col("valid_time") <= latest_valid)
        & (pl.col("valid_time") > refit - timedelta(days=WINDOW_DAYS))
        & (pl.col("power_fcst_init_time") <= refit)
    )
    too_new = known.filter(
        (pl.col("valid_time") > latest_valid) | (pl.col("power_fcst_init_time") > refit)
    ).height
    if known.is_empty() or too_new:
        msg = f"refit {refit}: no usable rows or a row from after the refit"
        raise ValueError(msg)
    return known


def arm_rows(*, known: pl.DataFrame, arm: Arm) -> pl.DataFrame:
    """Return an arm's rows, with whole groups of `arm.members` consecutive rows.

    Raises:
        ValueError: If a group does not hold `arm.members` rows.
    """
    rows = known.filter(pl.col("ensemble_member") == 0) if arm.members == 1 else known
    sizes = rows.group_by("power_fcst_init_time", "valid_time").len()["len"]
    if (sizes != arm.members).any():
        msg = f"arm {arm.name}: a group does not hold {arm.members} members"
        raise ValueError(msg)
    return rows


def start_margin(*, label: np.ndarray) -> np.ndarray:
    """Return the starting value of each quantile level: that quantile of the training labels."""
    return np.quantile(label, common.LEVELS).astype(np.float32)


def custom_objective(
    *, arm: Arm, weight: np.ndarray
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    """Return the gradient and Hessian function of a custom-objective arm.

    The gradient of each level is divided by twice its mass (CRPS) or kept as is (pinball), so that
    both arms have a gradient between -1 and 1 and the same constant Hessian.

    Args:
        arm: The arm; its `loss` is `crps` or `pinball_custom`.
        weight: Each row's recency weight.

    Returns:
        A function from (predictions, training matrix) to (gradient, Hessian), shape (rows, 11).
    """
    members = arm.members

    def function(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        label = dtrain.get_label().reshape(-1, objective.N_LEVELS)[:, 0]
        rows = predt.shape[0]
        atoms = predt.reshape(rows // members, members, objective.N_LEVELS)
        actual = label.reshape(rows // members, members)[:, 0]
        if arm.loss == "crps":
            gradient = objective.crps_gradient(
                atoms=atoms, actual=actual, members=members, fair=True
            ) / (2.0 * objective.CRPS_MASSES.astype(np.float32))
        else:
            gradient = objective.pinball_gradient(atoms=atoms, actual=actual)
        gradient = gradient.reshape(rows, objective.N_LEVELS) * weight[:, None]
        hessian = np.broadcast_to((weight * (HESSIAN / 2.0))[:, None], gradient.shape).astype(
            np.float32
        )
        return gradient, hessian

    return function


def fit_arm(
    *, arm: Arm, rows: pl.DataFrame, row: dict, refit: datetime, rounds: int, seed: int
) -> tuple[xgb.Booster, np.ndarray]:
    """Fit one arm for one series at one refit.

    Args:
        arm: The arm.
        rows: `arm_rows()` of the refit's known rows.
        row: The series' row of `common.series_table()`.
        refit: The refit time, which anchors the recency weights.
        rounds: Boosting rounds.
        seed: The fitting seed.

    Returns:
        The booster and the per-level starting values (float32, shape (11,)).
    """
    age_days = ((refit - rows["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    weight = (0.5 ** (age_days / HALF_LIFE_DAYS)).astype(np.float32)
    label = (rows["power"] / row["effective_capacity_mw"]).to_numpy().astype(np.float32)
    start = start_margin(label=label)
    margin = np.broadcast_to(start, (len(label), objective.N_LEVELS)).copy()
    matrix = rows.select(input_columns()).to_numpy().astype(np.float32)
    parameters: dict[str, object] = {**TREE_PARAMETERS, "eta": ETA, "seed": seed, "nthread": 1}
    if arm.loss == "pinball":
        parameters |= {"objective": "reg:quantileerror", "quantile_alpha": common.LEVELS}
        train = xgb.DMatrix(matrix, label=label, weight=weight, base_margin=margin)
        return xgb.train(parameters, train, num_boost_round=rounds), start
    train = xgb.DMatrix(
        matrix,
        label=np.repeat(label[:, None], objective.N_LEVELS, axis=1),
        base_margin=margin,
    )
    booster = xgb.train(
        parameters,
        train,
        num_boost_round=rounds,
        obj=custom_objective(arm=arm, weight=weight),
    )
    return booster, start


def predict_atoms(*, booster: xgb.Booster, matrix: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Return the 11 atoms of each row: the booster's margin from the starting values."""
    margin = np.broadcast_to(start, (len(matrix), objective.N_LEVELS)).copy()
    return booster.predict(xgb.DMatrix(matrix, base_margin=margin)).reshape(
        len(matrix), objective.N_LEVELS
    )


def mixture(*, atoms: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Pool the members' atoms into the mixture's 11 quantiles (as experiment 87 does).

    Args:
        atoms: Shape (cases, members, 11).
        masses: The mass of each of the 11 atoms of one member.

    Returns:
        Shape (cases, 11), float32, sorted along the levels.
    """
    cases, members, _levels = atoms.shape
    values = atoms.reshape(cases, members * objective.N_LEVELS)
    weights = np.tile(masses, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def prepared_run(*, run_rows: pl.DataFrame, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return one run's 51-member rows for one series with derived columns and lags, sorted.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = prepared.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    prepared = add_lags(frame=prepared, power=arrays.power).sort("valid_time", "ensemble_member")
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    return prepared


def served_runs() -> dict[datetime, list[datetime]]:
    """Return each monthly refit and the `power_fcst_init_time` of the gap runs it forecasts."""
    out: dict[datetime, list[datetime]] = {}
    for origin in common.origins_of_subset(subset="gap"):
        refit = max(r for r in MONTHLY if r <= origin)
        out.setdefault(refit, []).append(origin)
    return out


def job_path(*, arm: str, site: str, refit: datetime, root: Path = PREDS_DIR) -> Path:
    """Return the file one refit's predictions for one series and arm are written to."""
    return root / arm / f"{site}_{refit:%Y%m%dT%H%M}.parquet"


def run_site(
    *, site: str, arm_names: list[str], rounds: int, refits: int | None, root: Path, seed: int
) -> list[str]:
    """Fit every refit of one series for each arm and write the predictions.

    Args:
        site: The series label (D2, D3, ...).
        arm_names: Keys of `ARMS`.
        rounds: Boosting rounds.
        refits: Fit only this many of the first refits (for the smoke test), or `None` for all.
        root: The folder holding one subfolder per arm.
        seed: The fitting seed.

    Returns:
        One progress line per refit.
    """
    series = common.primary_series()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    arrays = common.load_series_arrays(series=series.filter(pl.col("site") == site))[site]
    all_rows = training_rows(row=row, arrays=arrays)
    lines = []
    for refit, origins in list(served_runs().items())[:refits]:
        wanted = [
            *arm_names,
            *(extra for extra, (source, _) in POOLED_AGAIN.items() if source in arm_names),
        ]
        todo = [
            a for a in wanted if not job_path(arm=a, site=site, refit=refit, root=root).exists()
        ]
        if not todo:
            continue
        sources = list(dict.fromkeys(POOLED_AGAIN[a][0] if a in POOLED_AGAIN else a for a in todo))
        started = time.monotonic()
        known = known_rows(rows=all_rows, refit=refit)
        models = {}
        for name in sources:
            arm = ARMS[name]
            fitted = time.monotonic()
            models[name] = fit_arm(
                arm=arm,
                rows=arm_rows(known=known, arm=arm),
                row=row,
                refit=refit,
                rounds=rounds,
                seed=seed,
            )
            lines.append(f"{site} {refit:%Y-%m} {name}: fit {time.monotonic() - fitted:.0f} s")
        pieces: dict[str, list[pl.DataFrame]] = {name: [] for name in todo}
        for origin in origins:
            frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
            prepared = prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                    *common.KEYS, "ensemble_member", *WEATHER
                ),
                row=row,
                arrays=arrays,
            )
            keys = prepared.select(*common.KEYS).unique(maintain_order=True)
            matrix = prepared.select(input_columns()).to_numpy().astype(np.float32)
            members = prepared["ensemble_member"].n_unique()
            for name in todo:
                source, masses = POOLED_AGAIN.get(name, (name, ARMS[name].masses))
                booster, start = models[source]
                atoms = predict_atoms(booster=booster, matrix=matrix, start=start)
                pooled = mixture(
                    atoms=atoms.reshape(-1, members, objective.N_LEVELS), masses=masses
                )
                pieces[name].append(
                    keys.hstack(
                        pl.DataFrame(pooled, schema=list(common.LEVEL_COLUMNS))
                    ).with_columns(seed=pl.lit(seed).cast(pl.Int8))
                )
        for name, tables in pieces.items():
            common.atomic_write(
                frame=pl.concat(tables),
                target=job_path(arm=name, site=site, refit=refit, root=root),
            )
        common.log_gpu_seconds(
            job=f"{site} {refit:%Y-%m} {','.join(todo)}",
            seconds=time.monotonic() - started,
            workers=1,
        )
    return lines


def _run(args: tuple[str, list[str], int, int | None, Path, int]) -> list[str]:
    site, arm_names, rounds, refits, root, seed = args
    return run_site(
        site=site, arm_names=arm_names, rounds=rounds, refits=refits, root=root, seed=seed
    )


def main() -> None:
    """Fit the chosen arms for the chosen series in parallel worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sites = args.sites or common.primary_series()["site"].to_list()
    root = PREDS_DIR.with_name("preds_smoke") if args.smoke else PREDS_DIR
    rounds = SMOKE_ROUNDS if args.smoke else ROUNDS
    refits = SMOKE_REFITS if args.smoke else None
    jobs = [(site, args.arms, rounds, refits, root, common.SEED) for site in sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for lines in pool.map(_run, jobs):
            for line in lines:
                print(line, flush=True)


if __name__ == "__main__":
    main()
