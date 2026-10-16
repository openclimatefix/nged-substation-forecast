"""Fit arms A, B, C, the placebo, and B3h of experiment 00057 and write their quantiles.

Every arm is an XGBoost nine-quantile model trained per solar farm and seed on the ENS control
member's rows of the 00 UTC runs of 2024-04-01 to 2025-06-30 (valid times 0 to 54 hours after the
run's 00 UTC start), with experiment 33's N1 recipe: recency weights with a half-life of 2 years,
the columns of the arm, `lead_hours`, and three lags of power from the forecast origin. Arm A has
the nine ENS solar columns; arm B adds three UKV columns (shortwave, total cloud, temperature);
arm C has the three UKV columns plus the solar zenith and azimuth; the placebo has arm B's columns
with the UKV columns shuffled among rows of the same farm, month, hour of day, and lead day; arm
B3h has arm B's columns rebuilt from UKV's steps at multiples of 3 hours. UKV is the CEDA
archive's 00 UTC run of the ENS run's date. At scoring, arms A, B, the placebo, and B3h run once per
ENS member and the 51 member quantile sets are pooled; arm C runs once.

Output: `work/arms/<arm>.parquet` (all seeds) for `a`, `b`, `c`, `b_placebo`, `b_3h`, and the two
reference arms `ref_best_guess` and `ref_n1`, all on the same cases, and `work/ukv_checks.json`.

    uv run python ml-experiments/00057-product-trained-xgboost-solar-farms/e00057_fit.py [--smoke]
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import e00057_common as common
import icechunk
import numpy as np
import polars as pl
import xgboost as xgb
import zarr
from e00057_common import KEY_COLUMNS, LEVEL_COLUMNS
from studies.grid_sampling import nearest_cells
from studies.ukv_ceda_profiles import STATUS_COMPLETE
from studies.ukv_ceda_stores import Store, make_stores, store_index

ARMS_DIR = common.WORK_DIR / "arms"
SEED_DIR = common.WORK_DIR / "seed_tables"
SLOT_CYCLE: int = 4
"""Four 6-hourly slots a day; the 00 UTC run is the first of the day."""
NATIVE = np.array(common.NATIVE_LEADS)
VARIABLES = ("shortwave_down", "cloud_total", "temperature_1p5m")


def zarr_array(*, group: zarr.Group, name: str) -> zarr.Array:
    """Return a member of the group that the store's layout guarantees is an array."""
    member = group[name]
    if not isinstance(member, zarr.Array):
        msg = f"{name} is not an array"
        raise TypeError(msg)
    return member


def open_stores() -> tuple[tuple[Store, ...], zarr.Group]:
    """Open the three UKV-CEDA stores read-only; return them and the first store's group."""
    groups, statuses = [], []
    for name in common.UKV_STORES:
        repository = icechunk.Repository.open(
            icechunk.local_filesystem_storage(str(common.NWP_DIR / name / "store"))
        )
        group = zarr.open_group(repository.readonly_session(branch="main").store, mode="r")
        groups.append(group)
        statuses.append(np.asarray(zarr_array(group=group, name="status")[:], dtype=np.int8))
    return make_stores(groups=groups, statuses=statuses), groups[0]


def farm_cells(*, group: zarr.Group, sites: pl.DataFrame) -> dict[int, int]:
    """Return each farm's nearest 2 km cell index; the coordinates are never printed."""
    cells = pl.DataFrame(
        {
            "cell_id": np.arange(zarr_array(group=group, name="cell_latitude").shape[0]),
            "latitude": np.asarray(zarr_array(group=group, name="cell_latitude")[:]),
            "longitude": np.asarray(zarr_array(group=group, name="cell_longitude")[:]),
        }
    )
    nearest = nearest_cells(sites=sites.select("site", "latitude", "longitude"), cells=cells)
    if float(nearest["distance_km"].to_numpy().max()) > 2.0:
        msg = "a farm lies more than 2 km from every UKV cell"
        raise ValueError(msg)
    by_site = dict(zip(nearest["site"], nearest["cell_id"], strict=True))
    return {int(r["time_series_id"]): int(by_site[r["site"]]) for r in sites.iter_rows(named=True)}


def fill_leads(*, values: np.ndarray) -> np.ndarray:
    """Rebuild every lead of a 55-lead series from its native leads by a straight line.

    Args:
        values: The series at the 55 leads, NaN where the store holds no value.

    Returns:
        The 55 leads, all NaN when any native lead (0 to 48, 51, 54) is not finite.
    """
    if not np.isfinite(values[NATIVE]).all():
        return np.full(common.UKV_LEADS, np.nan)
    return np.interp(np.arange(common.UKV_LEADS), NATIVE, values[NATIVE])


def run_features(
    *, shortwave: np.ndarray, cloud: np.ndarray, temperature: np.ndarray
) -> pl.DataFrame:
    """Return the UKV columns at every half-hour 0 to 54 hours after one run's start.

    The columns `ukv_*` use every native step, and the columns `ukv3h_*` use only the steps at
    multiples of 3 hours; both are joined linearly between steps, and a half-hour is the mean of
    the hours either side.
    """
    lead = np.arange(0, 2 * common.MAX_LEAD_HOURS + 1)  # half-hour steps
    low = lead // 2
    high = np.minimum((lead + 1) // 2, common.UKV_LEADS - 1)  # the same hour, or the next
    columns: dict[str, np.ndarray] = {"half_hours": lead}
    three_hourly = np.arange(0, common.UKV_LEADS, 3)
    names = ("shortwave", "cloud_total", "temperature")
    for name, series in zip(names, (shortwave, cloud, temperature), strict=True):
        filled = fill_leads(values=series)
        coarse = np.interp(np.arange(common.UKV_LEADS), three_hourly, filled[three_hourly])
        columns[f"ukv_{name}"] = filled[low] / 2 + filled[high] / 2
        columns[f"ukv3h_{name}"] = coarse[low] / 2 + coarse[high] / 2
    return pl.DataFrame(columns).filter(
        pl.all_horizontal(pl.col(*common.UKV_SOLAR, *common.UKV_SOLAR_3H).is_finite())
    )


def read_ukv(
    *, stores: tuple[Store, ...], cells: dict[int, int], runs: list[datetime]
) -> pl.DataFrame:
    """Return the UKV columns for every complete 00 UTC run in `runs` and every farm.

    Columns: `time_series_id`, `nwp_init_time` (the run's 00 UTC), `valid_time`, and the four UKV
    columns. A run that is not complete in its store is absent.
    """
    series = list(cells)
    indices = [cells[s] for s in series]
    pieces = []
    for run in runs:
        slot = SLOT_CYCLE * (run - common.GRID_START).days
        position = store_index(stores=stores, slot=slot)
        if position is None or stores[position].statuses[slot] != STATUS_COMPLETE:
            continue
        blocks = {
            name: np.asarray(stores[position].group[name][slot])[:, indices] for name in VARIABLES
        }
        for k, farm in enumerate(series):
            features = run_features(
                shortwave=blocks["shortwave_down"][:, k],
                cloud=blocks["cloud_total"][:, k],
                temperature=blocks["temperature_1p5m"][:, k],
            )
            pieces.append(
                features.with_columns(
                    time_series_id=pl.lit(farm, dtype=pl.Int32),
                    nwp_init_time=pl.lit(run, dtype=pl.Datetime("us", "UTC")),
                    valid_time=pl.lit(run, dtype=pl.Datetime("us", "UTC"))
                    + pl.duration(minutes=30) * pl.col("half_hours"),
                ).drop("half_hours")
            )
    return pl.concat(pieces)


def shuffled(*, ukv: pl.DataFrame) -> pl.DataFrame:
    """Move the UKV columns among rows of the same farm, month, hour of day, and lead day.

    Within a group the rows are cycled by one place in a random order, so no row keeps its own
    values.
    """
    rng = np.random.default_rng(common.PLACEBO_SEED)
    keyed = ukv.with_columns(
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        hour=pl.col("valid_time").dt.hour(),
        minute=pl.col("valid_time").dt.minute(),
        lead_day=(pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_hours() // 24,
    ).with_row_index("row")
    values = ukv.select(common.UKV_SOLAR).to_numpy().copy()
    out = values.copy()
    for part in keyed.partition_by("time_series_id", "month", "hour", "minute", "lead_day"):
        index = rng.permutation(part["row"].to_numpy())
        if index.size > 1:
            out[index] = values[np.roll(index, 1)]
    return ukv.drop(common.UKV_SOLAR).hstack(pl.DataFrame(out, schema=list(common.UKV_SOLAR)))


def train_rows(
    *, series: int, ukv: pl.DataFrame, power: np.ndarray, capacity: float
) -> pl.DataFrame:
    """Return one farm's training rows: ENS control member joined to UKV, with lags and target."""
    frame = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(
            (pl.col("time_series_id") == series)
            & (pl.col("ensemble_member") == 0)
            & (pl.col("valid_time") >= common.TRAIN_START)
            & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
            & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & pl.col("power").is_finite()
        )
        .select(*KEY_COLUMNS, "nwp_init_time", "power", *common.ENS_SOLAR)
        .collect()
    )
    frame = frame.filter(
        common.hours_after_run(
            valid_time=pl.col("valid_time"), power_fcst_init_time=pl.col("power_fcst_init_time")
        )
        <= common.MAX_LEAD_HOURS
    )
    joined = frame.join(
        ukv.filter(pl.col("time_series_id") == series).drop("time_series_id"),
        on=["nwp_init_time", "valid_time"],
        how="inner",
        validate="m:1",
    )
    return common.add_lags(frame=joined, power=power).with_columns(
        target=pl.col("power") / capacity
    )


def fit_arm(*, rows: pl.DataFrame, columns: tuple[str, ...], seed: int, rounds: int) -> xgb.Booster:
    """Fit one nine-level quantile model on `rows` with experiment 33's recency weights."""
    age_days = (common.LAST_TRAINING_VALID - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.train(
        {**common.XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            rows.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy(),
            label=rows["target"].to_numpy(),
            weight=0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS),
        ),
        num_boost_round=rounds,
    )


def predict(*, model: xgb.Booster, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Return the nine sorted quantiles of every row."""
    matrix = frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()
    return np.sort(model.predict(xgb.DMatrix(matrix)), axis=1)


def pooled(*, part: pl.DataFrame, member_quantiles: np.ndarray) -> pl.DataFrame:
    """Return the keys and nine mixture quantiles of every case of `part` (51 members each)."""
    ordered = (
        part.select(*KEY_COLUMNS, "ensemble_member")
        .with_columns(row=pl.int_range(pl.len()))
        .sort(*KEY_COLUMNS, "ensemble_member")
    )
    if ordered.height % common.MEMBERS:
        msg = f"a case lacks some of the {common.MEMBERS} members"
        raise ValueError(msg)
    grid = ordered["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
    if (grid != np.arange(common.MEMBERS)).any():
        msg = "the members of a case are not 0 to 50"
        raise ValueError(msg)
    matrix = member_quantiles[ordered["row"].to_numpy()].reshape(
        -1, common.MEMBERS, len(LEVEL_COLUMNS)
    )
    keys = ordered.gather_every(common.MEMBERS).select(KEY_COLUMNS)
    mixture = common.read_levels(
        fine=common.mixture_quantiles(member_quantiles=matrix.astype(np.float32))
    )
    return keys.hstack(pl.DataFrame(mixture, schema=list(LEVEL_COLUMNS)))


def validation_part(
    *, path: Any, series: int, ukv: pl.DataFrame, power: np.ndarray
) -> pl.DataFrame:
    """Return one run's ENS rows for a farm, with UKV, lags, and `lead_hours`, within 54 hours."""
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("time_series_id") == series)
        .select(*KEY_COLUMNS, "nwp_init_time", "ensemble_member", *common.ENS_SOLAR)
        .collect()
    )
    frame = frame.filter(
        common.hours_after_run(
            valid_time=pl.col("valid_time"), power_fcst_init_time=pl.col("power_fcst_init_time")
        )
        <= common.MAX_LEAD_HOURS
    ).join(
        ukv.filter(pl.col("time_series_id") == series).drop("time_series_id"),
        on=["nwp_init_time", "valid_time"],
        how="inner",
        validate="m:1",
    )
    return common.add_lags(frame=frame, power=power)


def merge_seeds(*, cases: pl.DataFrame, checks: dict[str, Any]) -> None:
    """Merge the tables each `--seed` process saved and write the arms."""
    checks["training_rows"] = json.loads((SEED_DIR / "training_rows.json").read_text())
    tables = {
        arm: [pl.read_parquet(path) for path in sorted(SEED_DIR.glob(f"{arm}_seed*.parquet"))]
        for arm in common.ARM_COLUMNS
    }
    write_all(tables=tables, cases=cases, checks=checks)


def main() -> None:  # noqa: PLR0915
    """Fit every arm for every seed and write the arm tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, no output")
    parser.add_argument("--seed", type=int, help="fit this seed only and save its tables")
    parser.add_argument("--nthread", type=int, default=6, help="XGBoost threads of this process")
    parser.add_argument("--merge", action="store_true", help="merge the saved seeds and write arms")
    args = parser.parse_args()
    smoke = args.smoke
    common.XGBOOST_PARAMETERS["nthread"] = args.nthread
    sites = common.site_table()
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in sites.iter_rows(named=True)
    }
    power = common.full_power(sites=sites)
    stores, first = open_stores()
    cells = farm_cells(group=first, sites=sites)
    cases = (
        pl.read_parquet(common.REFERENCE_PATHS["ref_best_guess"])
        .filter((pl.col("technology") == "solar") & (pl.col("seed") == 0))
        .select(*KEY_COLUMNS, "actual_norm")
    )
    runs_val = sorted(cases["power_fcst_init_time"].dt.truncate("1d").unique().to_list())
    runs_train = [
        common.TRAIN_START + timedelta(days=i)
        for i in range((common.LAST_TRAINING_VALID.date() - common.TRAIN_START.date()).days + 1)
    ]
    runs = runs_train + runs_val
    started = time.monotonic()
    ukv = read_ukv(stores=stores, cells=cells, runs=runs[:3] + runs[-2:] if smoke else runs)
    ukv_placebo = shuffled(ukv=ukv)
    have = set(ukv["nwp_init_time"].unique().to_list())
    checks: dict[str, Any] = {
        "training_runs_with_ukv": sum(r in have for r in runs_train),
        "training_runs": len(runs_train),
        "validation_runs_with_ukv": sum(r in have for r in runs_val),
        "validation_runs": len(runs_val),
        "validation_runs_without_ukv": [r.date().isoformat() for r in runs_val if r not in have],
        "ukv_mean_shortwave_w_m2": float(ukv["ukv_shortwave"].to_numpy().mean()),
        "ukv_read_seconds": round(time.monotonic() - started),
    }
    print(json.dumps(checks, indent=1), flush=True)
    paths = sorted(common.ENS_VALIDATION_DIR.glob("init=*.parquet"))
    paths = [
        p for p in paths if datetime.fromisoformat(p.stem.split("=")[1]).replace(tzinfo=UTC) in have
    ]
    if smoke:
        paths = paths[:2]
    tables: dict[str, list[pl.DataFrame]] = {arm: [] for arm in common.ARM_COLUMNS}
    seeds = common.SEEDS[:1] if smoke else common.SEEDS if args.seed is None else (args.seed,)
    if args.merge:
        merge_seeds(cases=cases, checks=checks)
        return
    for seed in seeds:
        for series in cells:
            rows = train_rows(
                series=series, ukv=ukv, power=power[series], capacity=capacity[series]
            )
            rows_placebo = train_rows(
                series=series, ukv=ukv_placebo, power=power[series], capacity=capacity[series]
            )
            checks.setdefault("training_rows", {})[str(series)] = rows.height
            models = {
                arm: fit_arm(
                    rows=rows_placebo if arm == "b_placebo" else rows,
                    columns=cols,
                    seed=seed,
                    rounds=20 if smoke else common.N_ROUNDS,
                )
                for arm, cols in common.ARM_COLUMNS.items()
            }
            for path in paths:
                for arm, cols in common.ARM_COLUMNS.items():
                    part = validation_part(
                        path=path,
                        series=series,
                        ukv=ukv_placebo if arm == "b_placebo" else ukv,
                        power=power[series],
                    ).sort(*KEY_COLUMNS, "ensemble_member")
                    if arm == "c":
                        part = part.filter(pl.col("ensemble_member") == 0)
                        keys = part.select(KEY_COLUMNS)
                        quantiles = predict(model=models[arm], frame=part, columns=cols)
                        table = keys.hstack(pl.DataFrame(quantiles, schema=list(LEVEL_COLUMNS)))
                    else:
                        table = pooled(
                            part=part,
                            member_quantiles=predict(model=models[arm], frame=part, columns=cols),
                        )
                    tables[arm].append(table.with_columns(seed=pl.lit(seed, dtype=pl.Int8)))
        print(f"seed {seed} done after {time.monotonic() - started:.0f} s", flush=True)
    if smoke:
        print("smoke run: no output written", flush=True)
        return
    if args.seed is not None:
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        (SEED_DIR / "training_rows.json").write_text(json.dumps(checks["training_rows"]))
        for arm, frames in tables.items():
            target = SEED_DIR / f"{arm}_seed{args.seed}.parquet"
            pl.concat(frames).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
        return
    write_all(tables=tables, cases=cases, checks=checks)


def write_all(
    *, tables: dict[str, list[pl.DataFrame]], cases: pl.DataFrame, checks: dict[str, Any]
) -> None:
    """Write every arm and the two reference arms on the cases all arms share."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    arms = {arm: pl.concat(frames) for arm, frames in tables.items()}
    shared = None
    for frame in arms.values():
        keys = frame.select("seed", *KEY_COLUMNS)
        shared = (
            keys if shared is None else shared.join(keys, on=["seed", *KEY_COLUMNS], how="inner")
        )
    assert shared is not None
    shared = shared.join(cases.select(*KEY_COLUMNS), on=list(KEY_COLUMNS), how="semi")
    actual = cases.select(*KEY_COLUMNS, "actual_norm")
    for arm, frame in arms.items():
        _write(name=arm, frame=frame, shared=shared, actual=actual)
    for name, path in common.REFERENCE_PATHS.items():
        reference = (
            pl.read_parquet(path)
            .filter(pl.col("technology") == "solar")
            .with_columns(seed=pl.col("seed").cast(pl.Int8))
            .select("seed", *KEY_COLUMNS, *LEVEL_COLUMNS)
        )
        _write(name=name, frame=reference, shared=shared, actual=actual)
    checks["cases_per_seed"] = shared.filter(pl.col("seed") == 0).height
    (common.WORK_DIR / "ukv_checks.json").write_text(json.dumps(checks, indent=1, default=str))


def _write(*, name: str, frame: pl.DataFrame, shared: pl.DataFrame, actual: pl.DataFrame) -> None:
    """Write one arm restricted to the shared cases, with the observation."""
    out = (
        shared.join(frame, on=["seed", *KEY_COLUMNS], how="left", validate="1:1")
        .join(actual, on=list(KEY_COLUMNS), how="left", validate="m:1")
        .sort("seed", *KEY_COLUMNS)
    )
    if out.select(pl.col(*LEVEL_COLUMNS).is_null().any()).to_numpy().any():
        msg = f"arm {name} lacks some shared case"
        raise ValueError(msg)
    target = ARMS_DIR / f"{name}.parquet"
    out.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)


if __name__ == "__main__":
    main()
