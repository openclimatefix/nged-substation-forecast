"""Stage 2 of experiment 00050: demand arms built as shifts of experiment 00046's `d1l` boosters.

No booster is refitted. Every arm is the same `d1l` demand model with the solar term laid over it,
and the arms differ only in the central anomaly the term uses:

- `members` predicts the nine quantiles of each of the 51 ENS members from experiment 00046's
  saved `d1l` boosters, once per seed and series (CPU, one thread per worker), and stores them
  under `work/member_quantiles/`.
- `shift` shifts member `m` by `-K(site, band) * C * (central + (a_m - mean(a))) / capacity`, where
  `C` is experiment 00040's 2023/24 capacity, `K` is experiment 00046's fitted lead-band
  coefficient (1 in bands where it fitted none), `central` is the arm's calibrated central anomaly
  (`solar/central_val.parquet`), and `a_m - mean(a)` is the raw ENS member's deviation from the ENS
  mean, added unscaled. The 51 shifted members are pooled into nine quantiles.
- `assemble` adds the 14-day correction and the 1.15 spread factor to each arm as experiment 00046
  did, joins the observed power, and writes the arms under `work/arms/` and as private prediction
  sets of experiment 50.

Arms (the first two exist whatever stage 1 selected): `ens` is experiment 00046's
`d1l_ens_shift_fitted` rebuilt, `best` is the arm stage 1 selected (written only when it is not
`ens`), and `gefs`, `blend`, and `blend_fixed` are the others. `shift` refuses to run until
`results/selection.json` exists.

    uv run python ml-experiments/00050-*/e00050_predict.py members [--workers 4]
    uv run python ml-experiments/00050-*/e00050_predict.py shift
    uv run python ml-experiments/00050-*/e00050_predict.py assemble
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00050_common as common
import numpy as np
import polars as pl
import xgboost as xgb

from studies import experiment_store

RADIATION_COLUMNS: Final[tuple[str, ...]] = ("shortwave", "longwave")
D1L_COLUMN_COUNT: Final[int] = 15
ARM_VARIANTS: Final[dict[str, str]] = {
    "ens": "central_ens",
    "gefs": "central_gefs",
    "blend": "central_blend",
    "blend_fixed": "central_blend_fixed",
}
"""Each shifted arm's name mapped to its column in `central_val.parquet`."""
VARIANT_SUFFIX: Final[str] = "shift_fitted"
BENCHMARK_ARM: Final[str] = "d1l_ens_shift_fitted"
TOLERANCE: Final[float] = 1e-5
MEMBER_FILE: Final[str] = "seed{seed}_{site}.parquet"


def arm_name(*, source: str) -> str:
    """Return the name of a shifted arm, such as `d1l_gefs_shift_fitted`."""
    return f"d1l_{source}_{VARIANT_SUFFIX}"


def input_columns() -> list[str]:
    """Return the `d1l` input columns in matrix order: experiment 00033's set without radiation.

    Returns:
        The column names.

    Raises:
        ValueError: If the count differs from 15.
    """
    columns = [
        c
        for c in (
            *common.feature_columns(calendar=True),
            *common.N1_EXTRA_COLUMNS,
            *common.LAG_COLUMNS,
        )
        if c not in RADIATION_COLUMNS
    ]
    if len(columns) != D1L_COLUMN_COUNT:
        msg = f"d1l has {len(columns)} columns, expected {D1L_COLUMN_COUNT}"
        raise ValueError(msg)
    return columns


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


def member_job(args: tuple[int, str]) -> tuple[str, float]:
    """Predict one seed and series' member quantiles for every validation run and write them."""
    seed, site = args
    target = common.MEMBER_DIR / MEMBER_FILE.format(seed=seed, site=site)
    if target.exists():
        return target.name, 0.0
    started = time.monotonic()
    series = common.series_table().filter(pl.col("site") == site)
    arrays = common.load_series_arrays(series=series)[site]
    row = next(series.iter_rows(named=True))
    booster = xgb.Booster()
    booster.load_model(common.BOOSTER_DIR / f"d1l_seed{seed}_{site}.ubj")
    booster.set_param({"device": "cpu", "nthread": 1})
    columns = input_columns()
    pieces = []
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
        run = add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        keys = run.select(*common.KEYS).unique(maintain_order=True)
        if run.height != keys.height * members or members != common.MEMBERS:
            msg = "every valid time must hold all 51 members"
            raise ValueError(msg)
        raw = booster.predict(xgb.DMatrix(run.select(columns).to_numpy().astype(np.float32)))
        quantiles = raw.reshape(-1, len(common.LEVELS)).astype(np.float32)
        pieces.append(
            run.select(*common.KEYS, "ensemble_member").hstack(
                pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS), orient="row")
            )
        )
    common.write_atomically(frame=pl.concat(pieces), target=target)
    return target.name, time.monotonic() - started


def run_members(*, workers: int, sites: list[str], seeds: list[int]) -> None:
    """Predict the member quantiles of every seed and series, `workers` jobs at a time."""
    jobs = [(seed, site) for seed in seeds for site in sites]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, seconds in pool.map(member_job, jobs):
            print(f"{name}: {seconds:.0f} s", flush=True)


def shifted_pool(*, member_quantiles: np.ndarray, member_shift: np.ndarray) -> np.ndarray:
    """Subtract each member's shift from its quantiles, then pool the members.

    Args:
        member_quantiles: Shape (cases, members, 9), float32.
        member_shift: Shape (cases, members), in the forecast's units (power over capacity).

    Returns:
        The pooled nine quantiles, shape (cases, 9). A shift of zero returns exactly the pooled
        unshifted quantiles.
    """
    shifted = member_quantiles - member_shift.astype(np.float32)[:, :, None]
    return common.mixture_quantiles(member_quantiles=shifted)


def member_shift(
    *,
    capacity_mw: float,
    effective_capacity_mw: float,
    coefficient: np.ndarray,
    central: np.ndarray,
    deviation: np.ndarray,
) -> np.ndarray:
    """Return each member's shift in power over capacity.

    Args:
        capacity_mw: The solar capacity `C` in MW.
        effective_capacity_mw: The series' effective capacity in MW.
        coefficient: `K` for each case, shape (cases,).
        central: The arm's calibrated central anomaly for each case, shape (cases,).
        deviation: Each member's raw ENS anomaly minus the ensemble mean, shape (cases, members).

    Returns:
        The shift, shape (cases, members); subtract it from the member's quantiles.
    """
    return (
        coefficient[:, None] * capacity_mw * (central[:, None] + deviation) / effective_capacity_mw
    )


def lead_band_coefficients(*, site: str) -> dict[int, float]:
    """Return experiment 00046's fitted multiplier for each lead band of one series."""
    return dict(
        pl.read_parquet(common.BAND_COEFFICIENTS_PATH)
        .filter(pl.col("site") == site)
        .select("band", "coefficient")
        .iter_rows()
    )


def run_deviation(*, run_file: pl.DataFrame, site: str, keys: pl.DataFrame) -> np.ndarray:
    """Return a run's raw ENS member deviations from the ensemble mean, shape (cases, members).

    Args:
        run_file: Experiment 00046's `ens_val` file of the run.
        site: The series label.
        keys: The run's distinct sorted keys, in case order.

    Returns:
        The deviations, NaN replaced by zero.

    Raises:
        ValueError: If the file's valid times differ from the run's.
    """
    own = run_file.filter(pl.col("site") == site).sort("valid_time", "ensemble_member")
    if not own["valid_time"].unique(maintain_order=True).equals(keys["valid_time"]):
        msg = f"the anomaly file of {site} does not cover the run's valid times"
        raise ValueError(msg)
    anomaly = own["anomaly"].to_numpy().astype(np.float64).reshape(keys.height, -1)
    anomaly = np.nan_to_num(anomaly, nan=0.0)
    return anomaly - anomaly.mean(axis=1, keepdims=True)


def central_vector(
    *, central: pl.DataFrame, site: str, keys: pl.DataFrame, column: str
) -> np.ndarray:
    """Return an arm's central anomaly at each of a run's cases, 0 where it is missing."""
    own = central.filter(pl.col("site") == site).select(
        "power_fcst_init_time", "valid_time", column
    )
    joined = keys.select("power_fcst_init_time", "valid_time").join(
        own, on=["power_fcst_init_time", "valid_time"], how="left", maintain_order="left"
    )
    return np.nan_to_num(joined[column].to_numpy().astype(np.float64), nan=0.0)


def shift_job(args: tuple[int, str]) -> tuple[str, float]:
    """Shift one seed and series' stored member quantiles under every arm and write the result."""
    seed, site = args
    target = common.RAW_ARMS_DIR / f"seed{seed}_{site}.parquet"
    if target.exists():
        return target.name, 0.0
    started = time.monotonic()
    row = next(common.series_table().filter(pl.col("site") == site).iter_rows(named=True))
    capacity = {r["site"]: r["C"] for r in common.capacities().iter_rows(named=True)}[site]
    band_k = lead_band_coefficients(site=site)
    central = pl.read_parquet(common.SOLAR_DIR / "central_val.parquet")
    members = pl.read_parquet(common.MEMBER_DIR / MEMBER_FILE.format(seed=seed, site=site))
    pieces = []
    for path in sorted((common.SOURCE_SOLAR_DIR / "ens_val").glob("init=*.parquet")):
        run_file = pl.read_parquet(path)
        origin = run_file["power_fcst_init_time"][0]
        part = members.filter(pl.col("power_fcst_init_time") == origin).sort(
            "valid_time", "ensemble_member"
        )
        keys = part.select(*common.KEYS).unique(maintain_order=True)
        quantiles = (
            part.select(common.LEVEL_COLUMNS).to_numpy().reshape(keys.height, common.MEMBERS, -1)
        )
        deviation = run_deviation(run_file=run_file, site=site, keys=keys)
        lead = common.lead_hours_of(frame=keys)
        coefficient = np.array([band_k.get(int(b), 1.0) for b in common.lead_band(lead_hours=lead)])
        for source, column in ARM_VARIANTS.items():
            shift = member_shift(
                capacity_mw=capacity,
                effective_capacity_mw=row["effective_capacity_mw"],
                coefficient=coefficient,
                central=central_vector(central=central, site=site, keys=keys, column=column),
                deviation=deviation,
            )
            pooled = shifted_pool(member_quantiles=quantiles, member_shift=shift)
            pieces.append(
                keys.with_columns(
                    variant=pl.lit(arm_name(source=source)), seed=pl.lit(seed).cast(pl.Int8)
                ).hstack(pl.DataFrame(pooled.astype(np.float32), schema=list(common.LEVEL_COLUMNS)))
            )
    common.write_atomically(frame=pl.concat(pieces), target=target)
    return target.name, time.monotonic() - started


def require_selection() -> str:
    """Return the arm stage 1 selected, or raise when stage 1 has not run."""
    if not common.SELECTION_PATH.exists():
        msg = f"{common.SELECTION_PATH} is missing; run e00050_anomaly.py cv first"
        raise FileNotFoundError(msg)
    return str(json.loads(common.SELECTION_PATH.read_text())["selected"])


def run_shift(*, workers: int, sites: list[str], seeds: list[int]) -> None:
    """Shift the stored member quantiles under every arm, `workers` jobs at a time."""
    selected = require_selection()
    print(f"stage 1 selected: {selected}")
    jobs = [(seed, site) for seed in seeds for site in sites]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, seconds in pool.map(shift_job, jobs):
            print(f"{name}: {seconds:.0f} s", flush=True)


def raw_variant(*, variant: str) -> pl.DataFrame:
    """Return one shifted arm's raw quantiles for every seed and series."""
    pieces = []
    for seed in common.SEEDS:
        for site in common.SITES:
            path = common.RAW_ARMS_DIR / f"seed{seed}_{site}.parquet"
            if not path.exists():
                msg = f"{path} is missing"
                raise FileNotFoundError(msg)
            pieces.append(
                pl.read_parquet(path)
                .filter(pl.col("variant") == variant)
                .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
                .with_columns(pl.col("seed").cast(pl.Int8))
            )
    return pl.concat(pieces)


def run_assemble() -> None:
    """Write each arm's scored form and prediction set, then check the benchmark against 00046's."""
    selected = require_selection()
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    names = {source: arm_name(source=source) for source in ARM_VARIANTS}
    if selected != "ens":
        names["best"] = "d1l_best_shift_fitted"
    for source, name in names.items():
        target = common.ARMS_DIR / f"{name}.parquet"
        if target.exists():
            continue
        variant = arm_name(source=selected if source == "best" else source)
        corrected = common.with_corrections(
            base=raw_variant(variant=variant), series=series, arrays=arrays
        )
        scored = common.stack(frame=corrected, correct=True, spread=True).filter(
            pl.col("valid_time") < common.HOLDOUT
        )
        common.write_atomically(
            frame=scored.join(actual, on=list(common.KEYS), how="left", validate="m:1"),
            target=target,
        )
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=name,
            frame=scored,
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        print(f"{name}: {scored.height:,} rows", flush=True)
    check_benchmark()


def check_benchmark() -> None:
    """Compare the rebuilt benchmark arm with experiment 00046's stored `d1l_ens_shift_fitted`.

    Raises:
        AssertionError: If any quantile differs by more than `TOLERANCE`, or the keys differ.
    """
    mine = pl.read_parquet(common.ARMS_DIR / f"{BENCHMARK_ARM}.parquet")
    theirs = pl.read_parquet(common.SOURCE_ARMS_DIR / f"{BENCHMARK_ARM}.parquet")
    keys = [*common.KEYS, "seed"]
    both = mine.join(theirs, on=keys, how="inner", suffix="_source", validate="1:1")
    worst = max(
        common.scalar(value=(both[c] - both[f"{c}_source"]).abs().max())
        for c in common.LEVEL_COLUMNS
    )
    print(
        f"check 1: rebuilt {BENCHMARK_ARM}: {both.height:,} of {mine.height:,} rows match "
        f"experiment 00046's {theirs.height:,}; largest quantile difference {worst:.2e}"
    )
    assert mine.height == theirs.height == both.height, "the benchmark's keys differ"
    assert worst <= TOLERANCE, "the rebuilt benchmark differs from experiment 00046's"


def main() -> None:
    """Run the requested step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("members", "shift", "assemble"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(common.SITES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.SEEDS))
    args = parser.parse_args()
    if args.step == "members":
        run_members(workers=args.workers, sites=args.sites, seeds=args.seeds)
    elif args.step == "shift":
        run_shift(workers=args.workers, sites=args.sites, seeds=args.seeds)
    else:
        run_assemble()


if __name__ == "__main__":
    main()
