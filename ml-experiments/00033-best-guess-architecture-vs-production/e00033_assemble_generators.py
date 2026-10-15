"""Assemble the generator arms of experiment 00033 into one table per arm.

Reads the stored prediction sets of experiments 00020, 00026, and 00029, the stored production,
`manual_heuristic`, and `climatology` forecasts of the `power_forecasts` Delta table (the quantiles
of each forecast's members at the nine levels, divided by the series' effective capacity), and the
new arms of `e00033_fit_generators.py` (N1 for generators and the ERA5-weather ceiling arms, each
also with the spread factor fitted month by month on earlier cases, as experiment 00020 did).

Output: `work/generator_tables/<arm>.parquet` with the run keys, `seed`, `lead_hours`,
`actual_norm`, `technology`, `band`, the nine quantile columns, and `members`.

    uv run python ml-experiments/00033-*/e00033_assemble_generators.py
"""

from pathlib import Path
from typing import Final

import e00033_common as demand
import numpy as np
import polars as pl
from e00033_fit_generators import ARM_NAMES
from e00033_gen_common import (
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    MIN_FIT_CASES,
    SHORT_BANDS,
    TRIVIAL_LEVEL,
    UPPER_INDEX,
    fit_spread_factor,
    scale_about_median,
    site_table,
    validation_cases,
)

from studies import experiment_store

OUT_DIR: Final[Path] = demand.WORK_DIR / "generator_tables"
POWER_FORECASTS: Final[Path] = demand.ML_EXPERIMENTS_DATA_DIR.parent / "power_forecasts"
FOLD_ID: Final[str] = "mid_2025_to_mid_2026"
TELEMETRY_DELAY: Final = np.timedelta64(210, "m")
"""A case is known to the spread-factor fit 3.5 hours after its valid time."""
STORED_BASELINES: Final[dict[str, str]] = {
    "production": "xgboost_baseline",
    "production_retrain_638": "xgboost_baseline_retrain_638",
    "production_retrain_790": "xgboost_baseline_retrain_790",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
EARLIER: Final[dict[str, tuple[int, str]]] = {
    "bg_base": (20, "base"),
    "bg_base_spread": (20, "base-spread"),
    "bg_margin": (20, "margin"),
    "bg_full": (20, "full"),
    "cams_spread": (20, "full-no-s0"),
    "cams_plain": (20, "cams"),
    "capacity_a0": (26, "a0"),
    "capacity_a4": (26, "a4"),
    "physics_scipy": (29, "scipy"),
    "physics_hybrid": (29, "torch-hybrid"),
}
NEW_ARMS: Final[tuple[str, ...]] = (
    "n1",
    "pw_era5",
    "pw_cams",
    "pw_sw_era5",
    "pw_sw_cams",
    "pw_ws100",
)
SPREAD_ARMS: Final[tuple[str, ...]] = ("pw_era5", "pw_cams", "pw_sw_era5", "pw_sw_cams", "pw_ws100")


def apply_spread(*, cases: pl.DataFrame, mixture: np.ndarray, technology: str) -> np.ndarray:
    """Return the quantiles after the spread factor, fitted month by month on earlier cases."""
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    short = np.isin(cases["band"].to_numpy(), list(SHORT_BANDS))
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out = mixture.copy()
    groups = [np.ones_like(short)] if technology == "solar" else [short, ~short]
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for group in groups:
            target = in_month & group & scorable
            fit_index = np.flatnonzero((valid <= cutoff - TELEMETRY_DELAY) & group & scorable)
            if not target.any() or fit_index.size < MIN_FIT_CASES:
                continue
            factor = fit_spread_factor(quantiles=mixture[fit_index], actual=actual[fit_index])
            out[target] = scale_about_median(quantiles=mixture[target], shift=0.0, factor=factor)
    return out


def stored_quantiles(*, experiment_name: str, sites: pl.DataFrame, inits: list) -> pl.DataFrame:
    """Return one stored baseline's capacity-normalised member quantiles for the generators."""
    capacity = pl.DataFrame(
        {
            "time_series_id": sites["time_series_id"],
            "capacity": sites["effective_capacity_mw"].cast(pl.Float64),
        }
    ).lazy()
    frame = (
        pl.scan_delta(str(POWER_FORECASTS))
        .filter(
            (pl.col("experiment_name") == experiment_name)
            & (pl.col("fold_id") == FOLD_ID)
            & pl.col("power_fcst_init_time").is_in(inits)
            & pl.col("time_series_id").is_in(sites["time_series_id"].to_list())
        )
        .group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(
            *[
                pl.col("power_fcst").cast(pl.Float64).quantile(float(level), "linear").alias(name)
                for level, name in zip(demand.LEVELS, demand.LEVEL_COLUMNS, strict=True)
            ],
            members=pl.col("ensemble_member").n_unique(),
            stored_rows=pl.len(),
        )
        .join(capacity, on="time_series_id", how="left", validate="m:1")
        .with_columns(*[pl.col(name) / pl.col("capacity") for name in demand.LEVEL_COLUMNS])
        .drop("capacity")
        .collect(engine="streaming")
    )
    if frame.filter(pl.col("members") != pl.col("stored_rows")).height:
        msg = f"{experiment_name}: a case holds duplicate member rows"
        raise ValueError(msg)
    return frame.drop("stored_rows")


def write(*, name: str, frame: pl.DataFrame) -> None:
    """Write one arm's table."""
    target = OUT_DIR / f"{name}.parquet"
    frame.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)


def assemble_new_arm(*, arm: str, cases: pl.DataFrame) -> None:
    """Write a new fitted arm's table, and its spread-factor table where the arm has one."""
    pieces, spread_pieces = [], []
    for technology in ("solar", "wind"):
        if arm not in ARM_NAMES[technology]:
            continue
        technology_cases = cases.filter(pl.col("technology") == technology)
        for seed in demand.SEEDS:
            path = demand.GENERATOR_ARMS_DIR / f"{arm}_{technology}_seed{seed}.parquet"
            if not path.exists():
                msg = f"{path.name} is missing; run e00033_fit_generators.py to the end first"
                raise FileNotFoundError(msg)
            joined = technology_cases.join(
                pl.read_parquet(path), on=list(KEY_COLUMNS), how="left", validate="1:1"
            )
            if joined[LEVEL_COLUMNS[0]].null_count():
                msg = f"{arm} {technology} seed {seed}: some cases have no forecast"
                raise ValueError(msg)
            pieces.append(joined.with_columns(seed=pl.lit(seed)))
            if arm in SPREAD_ARMS:
                spread = apply_spread(
                    cases=joined,
                    mixture=joined.select(LEVEL_COLUMNS).to_numpy().astype(np.float64),
                    technology=technology,
                )
                spread_pieces.append(
                    joined.drop(LEVEL_COLUMNS)
                    .hstack(pl.DataFrame(spread.astype(np.float32), schema=list(LEVEL_COLUMNS)))
                    .with_columns(seed=pl.lit(seed))
                )
    if pieces:
        write(name=arm, frame=pl.concat(pieces))
        print(f"{arm}: {sum(p.height for p in pieces):,} rows", flush=True)
    if spread_pieces:
        write(name=f"{arm}_spread", frame=pl.concat(spread_pieces))


def main() -> None:
    """Write every generator arm's table."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = validation_cases()
    sites = site_table()
    base_columns = [*KEY_COLUMNS, "lead_hours", "actual_norm", "technology", "band"]
    inits = cases["power_fcst_init_time"].unique().sort().to_list()
    for name, (experiment_id, stored) in EARLIER.items():
        frame = experiment_store.read_predictions(
            experiment_id=experiment_id, name=stored, row_set_id=demand.GENERATOR_ROW_SET_ID
        )
        keep = [*base_columns, "seed", *LEVEL_COLUMNS]
        if "band" not in frame.columns:
            frame = frame.join(
                cases.select(*KEY_COLUMNS, "band"), on=list(KEY_COLUMNS), how="left", validate="m:1"
            )
        if "technology" not in frame.columns:
            frame = frame.join(
                cases.select("time_series_id", "technology").unique(),
                on="time_series_id",
                how="left",
                validate="m:1",
            )
        write(name=name, frame=frame.select(keep))
        print(f"{name}: {frame.height:,} rows", flush=True)
    for name, experiment_name in STORED_BASELINES.items():
        stored = stored_quantiles(experiment_name=experiment_name, sites=sites, inits=inits)
        joined = cases.join(stored, on=list(KEY_COLUMNS), how="left", validate="1:1")
        missing = int(joined[LEVEL_COLUMNS[0]].null_count())
        frame = demand.repeat_seeds(frame=joined.select(*base_columns, *LEVEL_COLUMNS, "members"))
        write(name=name, frame=frame)
        fewest = int(joined["members"].min())
        print(
            f"{name}: {frame.height:,} rows, {missing} cases without a stored forecast, "
            f"fewest members {fewest}",
            flush=True,
        )
    for arm in NEW_ARMS:
        assemble_new_arm(arm=arm, cases=cases)


if __name__ == "__main__":
    main()
