"""Fit arms A, B, C (and the placebo) of experiment 00054 and write their quantiles.

Every arm is an XGBoost nine-quantile model trained per farm and seed on the ENS control member's
rows of the 00 UTC runs of 2024-04-01 to 2025-06-30 (valid times 0 to 54 hours after the run's
00 UTC start), with experiment 33's N1 recipe: recency weights with a half-life of 2 years, the
columns of the arm, `lead_hours`, and three lags of power from the forecast origin. Arm A has the
six ENS wind columns; arm B adds four UKV columns; arm C has the four UKV columns alone; the
placebo has arm B's columns with the UKV columns shuffled among rows of the same farm, month, hour
of day, and lead day. UKV is the CEDA archive's 00 UTC run of the ENS run's date. At scoring, arms
A and B run once per ENS member and the 51 member quantile sets are pooled; arm C runs once.

Output: `work/arms/<arm>.parquet` (all seeds) for `a`, `b`, `c`, `b_placebo`, and the two reference
arms `ref_best_guess` and `ref_n1`, all on the same cases, and `work/ukv_checks.json`.

    uv run python ml-experiments/00054-product-trained-xgboost-wind-farms/e00054_fit.py [--smoke]
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import e00054_common as common
import icechunk
import numpy as np
import polars as pl
import xgboost as xgb
import zarr
from e00054_common import KEY_COLUMNS, LEVEL_COLUMNS
from studies.grid_sampling import nearest_cells
from studies.ukv_ceda_profiles import STATUS_COMPLETE
from studies.ukv_ceda_stores import Store, make_stores, store_index

ARMS_DIR = common.WORK_DIR / "arms"
SLOT_CYCLE: int = 4
"""Four 6-hourly slots a day; the 00 UTC run is the first of the day."""
NATIVE = np.array(common.NATIVE_LEADS)
VARIABLES = ("wind_speed_10m", "wind_direction_10m", "wind_speed_925hpa")


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

    Returns:
        The 55 leads, all NaN when any native lead (0 to 48, 51, 54) is not finite.
    """
    if not np.isfinite(values[NATIVE]).all():
        return np.full(common.UKV_LEADS, np.nan)
    return np.interp(np.arange(common.UKV_LEADS), NATIVE, values[NATIVE])


def run_features(*, speed10: np.ndarray, dir10: np.ndarray, speed925: np.ndarray) -> pl.DataFrame:
    """Return the four UKV columns at every half-hour 0 to 54 hours after one run's start."""
    u = fill_leads(values=-speed10 * np.sin(np.radians(dir10)))
    v = fill_leads(values=-speed10 * np.cos(np.radians(dir10)))
    s925 = fill_leads(values=speed925)
    lead = np.arange(0, 2 * common.MAX_LEAD_HOURS + 1)  # half-hour steps
    low = lead // 2
    high = np.minimum((lead + 1) // 2, common.UKV_LEADS - 1)  # the same hour, or the next
    u_t, v_t, s_t = (x[low] / 2 + x[high] / 2 for x in (u, v, s925))
    bearing = np.radians((np.degrees(np.arctan2(u_t, v_t)) + 180.0) % 360.0)
    return pl.DataFrame(
        {
            "half_hours": lead,
            "ukv_speed_10m": np.hypot(u_t, v_t),
            "ukv_sin_10m": np.sin(bearing),
            "ukv_cos_10m": np.cos(bearing),
            "ukv_speed_925hpa": s_t,
        }
    ).filter(pl.all_horizontal(pl.col(*common.UKV_WIND).is_finite()))


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
                speed10=blocks["wind_speed_10m"][:, k],
                dir10=blocks["wind_direction_10m"][:, k],
                speed925=blocks["wind_speed_925hpa"][:, k],
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
    values = ukv.select(common.UKV_WIND).to_numpy().copy()
    out = values.copy()
    for part in keyed.partition_by("time_series_id", "month", "hour", "minute", "lead_day"):
        index = rng.permutation(part["row"].to_numpy())
        if index.size > 1:
            out[index] = values[np.roll(index, 1)]
    return ukv.select(*KEY_COLUMNS[:0], "time_series_id", "nwp_init_time", "valid_time").hstack(
        pl.DataFrame(out, schema=list(common.UKV_WIND))
    )


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
        .select(*KEY_COLUMNS, "nwp_init_time", "power", *common.ENS_WIND)
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
        .select(*KEY_COLUMNS, "nwp_init_time", "ensemble_member", *common.ENS_WIND)
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


def main() -> None:
    """Fit every arm for every seed and write the arm tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, no output")
    smoke = parser.parse_args().smoke
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
        .filter((pl.col("technology") == "wind") & (pl.col("seed") == 0))
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
        "ukv_mean_speed_10m_m_s": float(ukv["ukv_speed_10m"].to_numpy().mean()),
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
    for seed in common.SEEDS[:1] if smoke else common.SEEDS:
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
            .filter(pl.col("technology") == "wind")
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
