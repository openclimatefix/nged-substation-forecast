"""Fit the demand arms of experiment 00033 that no earlier experiment stored.

Three arms, each written as the 51-member (or single-input) mixture's nine quantiles per case:

- `era5_retrained`: N1 (half-life 2 years) retrained on ERA5 weather at the valid times of the ENS
  control-member training rows, and fed ERA5 at the valid time (one input, so the forecast is the
  model's nine quantiles). All 20 series, seeds 0 to 2.
- `cams_shortwave`: experiment 00022's N1H (half-life 0.5 years, trained on ENS) fed ERA5 at the
  valid time with `shortwave` replaced by CAMS at the series' own location. All 20 series, seeds 0
  to 2.
- `monthly`: each Primary substation's N1 refitted on the first day of each month from 2025-07 to
  2026-06 (half-life 0.5 years anchored at the refit date) on the ENS control member of every daily
  run whose valid time is at least 3.5 hours before the refit, and applied to the 51 ENS members of
  the validation runs whose origin falls in the month. Seed 0 only.

Output: `work/arms/<arm>_seed<s>.parquet`, keyed by series, run, and valid time. Existing files
stay.

    uv run python ml-experiments/00033-*/e00033_fit_demand.py [--smoke] [--only ARM]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00033_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import STUDIES_DATA_DIR

ARMS_DIR: Final[Path] = common.ARMS_DIR
AFTER_DIR: Final[Path] = common.WORK_DIR / "ens_after_training"
CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR / "downloads" / "reanalysis" / "CAMS" / "demand_points"
) / "cams_demand_points.parquet"
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
MONTH_STARTS: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve refit dates, 2025-07-01 to 2026-06-01."""
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
KEY_SEED: Final[tuple[str, ...]] = (*common.KEYS, "seed")


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


def train_matrix(
    *, rows: pl.DataFrame, row: dict, anchor: datetime, half_life_days: float
) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows (derived, lags, and weather set)."""
    age_days = (anchor - rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.DMatrix(
        rows.select(input_columns()).to_numpy().astype(np.float32),
        label=(rows["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days.to_numpy() / half_life_days),
    )


def own_rows(*, row: dict, frame: pl.DataFrame, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return one series' finite-power rows with derived columns and lags."""
    own = frame.filter(pl.col("time_series_id") == row["time_series_id"])
    own = own.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    own = common.with_derived(frame=own, latitude=row["latitude"], longitude=row["longitude"])
    own = own.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=own, power=arrays.power)


def fit(*, matrix: xgb.DMatrix, seed: int) -> xgb.Booster:
    """Fit one quantile model."""
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed},
        matrix,
        num_boost_round=common.N_ROUNDS,
    )


def era5_inputs(
    *, run: pl.DataFrame, arrays: common.SeriesArrays, cams: np.ndarray | None
) -> pl.DataFrame:
    """Return a run's single-member rows with ERA5 (and CAMS shortwave) at the valid time."""
    keys = run.select(*common.KEYS).unique(maintain_order=True).sort("valid_time")
    index = np.asarray(common.grid_index(times=keys["valid_time"]))
    names = common.feature_columns(calendar=True)
    era5 = arrays.features[index]
    swapped = keys.with_columns(
        *[pl.Series(c, era5[:, names.index(c)]) for c in WEATHER if c in names]
    )
    if cams is not None:
        swapped = swapped.with_columns(shortwave=pl.Series(cams[index].astype(np.float32)))
    return swapped


def predict_single(
    *, row: dict, arrays: common.SeriesArrays, model: xgb.Booster, cams: np.ndarray | None
) -> pl.DataFrame:
    """Return the nine quantiles of every validation case fed one ERA5 input."""
    pieces = []
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        inputs = era5_inputs(run=run, arrays=arrays, cams=cams)
        inputs = common.with_derived(
            frame=inputs, latitude=row["latitude"], longitude=row["longitude"]
        )
        inputs = add_lags(frame=inputs, power=arrays.power).with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        raw = model.predict(
            xgb.DMatrix(inputs.select(input_columns()).to_numpy().astype(np.float32))
        )
        pieces.append(
            inputs.select(*common.KEYS).hstack(
                pl.DataFrame(
                    np.sort(raw, axis=1).astype(np.float32), schema=list(common.LEVEL_COLUMNS)
                )
            )
        )
    return pl.concat(pieces)


def predict_members(
    *, row: dict, arrays: common.SeriesArrays, models: dict[datetime, xgb.Booster]
) -> pl.DataFrame:
    """Return the mixture quantiles of every validation run, from the latest refit by origin."""
    refits = sorted(models)
    pieces = []
    columns = input_columns()
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        origin = run["power_fcst_init_time"][0]
        chosen = max(d for d in refits if d <= origin)
        run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
        run = add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        if run.height != run.select(*common.KEYS).n_unique() * members:
            msg = "every valid time must hold every member"
            raise ValueError(msg)
        raw = (
            models[chosen]
            .predict(xgb.DMatrix(run.select(columns).to_numpy().astype(np.float32)))
            .reshape(-1, members, len(common.LEVELS))
        )
        pieces.append(
            run.select(*common.KEYS)
            .unique(maintain_order=True)
            .hstack(
                pl.DataFrame(
                    mixture_quantiles(member_quantiles=raw), schema=list(common.LEVEL_COLUMNS)
                )
            )
        )
    return pl.concat(pieces)


def cams_array(*, site: str) -> np.ndarray:
    """Return the series' CAMS irradiance on the dense grid (hourly means spread to half-hours)."""
    hourly = pl.read_parquet(CAMS_PATH).filter(pl.col("site") == site).select("time", "ghi_w_m2")
    joined = (
        pl.DataFrame({"valid_time": common.grid_times()})
        .with_columns(
            hour=(pl.col("valid_time") - pl.duration(microseconds=1)).dt.truncate("1h")
            + pl.duration(hours=1)
        )
        .join(hourly, left_on="hour", right_on="time", how="left", validate="m:1")
    )
    return joined["ghi_w_m2"].to_numpy().astype(np.float64)


def write(*, arm: str, seed: int, pieces: list[pl.DataFrame], smoke: bool) -> None:
    """Write one arm and seed."""
    out = ARMS_DIR.with_name("arms_smoke") if smoke else ARMS_DIR
    out.mkdir(parents=True, exist_ok=True)
    frame = pl.concat(pieces).with_columns(seed=pl.lit(seed).cast(pl.Int8))
    target = out / f"{arm}_seed{seed}.parquet"
    frame.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)


def monthly_models(
    *, row: dict, arrays: common.SeriesArrays, combined: pl.DataFrame, seed: int
) -> dict[datetime, xgb.Booster]:
    """Refit one series on each refit date from the rows known at least 3.5 hours earlier."""
    rows = own_rows(row=row, frame=combined, arrays=arrays)
    models = {}
    for refit in MONTH_STARTS:
        known = rows.filter(pl.col("valid_time") <= refit - REFIT_CUTOFF_AGE)
        known = known.filter(pl.col("power_fcst_init_time") <= refit)
        models[refit] = fit(
            matrix=train_matrix(
                rows=known, row=row, anchor=refit, half_life_days=common.SHORT_HALF_LIFE_DAYS
            ),
            seed=seed,
        )
    return models


def combined_rows() -> pl.DataFrame:
    """Return the ENS control-member rows before and after 2025-06-30 for the monthly refits."""
    before = pl.read_parquet(common.ENS_TRAIN_PATH).select(
        *common.KEYS, "ensemble_member", "power", *WEATHER
    )
    after = pl.concat([pl.read_parquet(p) for p in sorted(AFTER_DIR.glob("init=*.parquet"))])
    padded = after.with_columns(power=pl.lit(None, dtype=pl.Float32)).select(before.columns)
    return pl.concat([before, padded.cast(before.schema)])


def fill_after_power(
    *, frame: pl.DataFrame, arrays: dict[str, common.SeriesArrays], series: pl.DataFrame
) -> pl.DataFrame:
    """Fill the observed power of rows that carry none, from the dense grid."""
    pieces = []
    for row in series.iter_rows(named=True):
        part = frame.filter(pl.col("time_series_id") == row["time_series_id"])
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        filled = (
            common.take(values=arrays[row["site"]].power, index=index)
            * row["effective_capacity_mw"]
        )
        pieces.append(
            part.with_columns(
                power=pl.when(pl.col("power").is_null())
                .then(pl.Series(filled))
                .otherwise(pl.col("power"))
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Fit and write the requested arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="two series, one seed, scratch folder")
    parser.add_argument("--only", choices=("era5_retrained", "cams_shortwave", "monthly"))
    args = parser.parse_args()
    series = common.series_table()
    if args.smoke:
        series = series.head(2)
    arrays = common.load_series_arrays(series=series)
    train = pl.read_parquet(common.ENS_TRAIN_PATH)
    seeds = common.SEEDS[:1] if args.smoke else common.SEEDS
    out = ARMS_DIR.with_name("arms_smoke") if args.smoke else ARMS_DIR
    for arm in ("era5_retrained", "cams_shortwave"):
        if args.only not in (None, arm):
            continue
        for seed in seeds:
            if (out / f"{arm}_seed{seed}.parquet").exists():
                continue
            pieces = []
            for row in series.iter_rows(named=True):
                started = time.monotonic()
                site = arrays[row["site"]]
                rows = own_rows(
                    row=row,
                    frame=train.filter(pl.col("valid_time") <= common.LAST_TRAINING_VALID),
                    arrays=site,
                )
                if arm == "era5_retrained":
                    index = np.asarray(common.grid_index(times=rows["valid_time"]))
                    names = common.feature_columns(calendar=True)
                    rows = rows.with_columns(
                        *[pl.Series(c, site.features[index, names.index(c)]) for c in WEATHER]
                    )
                    matrix = train_matrix(
                        rows=rows,
                        row=row,
                        anchor=common.LAST_TRAINING_VALID,
                        half_life_days=common.RECENCY_HALF_LIFE_DAYS,
                    )
                    cams = None
                else:
                    matrix = train_matrix(
                        rows=rows,
                        row=row,
                        anchor=common.LAST_TRAINING_VALID,
                        half_life_days=common.SHORT_HALF_LIFE_DAYS,
                    )
                    cams = cams_array(site=row["site"])
                model = fit(matrix=matrix, seed=seed)
                pieces.append(predict_single(row=row, arrays=site, model=model, cams=cams))
                print(
                    f"{arm} seed {seed} {row['site']}: {time.monotonic() - started:.0f} s",
                    flush=True,
                )
            write(arm=arm, seed=seed, pieces=pieces, smoke=args.smoke)
    if args.only in (None, "monthly") and not (out / "monthly_seed0.parquet").exists():
        primary = series.filter(pl.col("substation_type") == "Primary")
        combined = fill_after_power(frame=combined_rows(), arrays=arrays, series=primary)
        pieces = []
        for row in primary.iter_rows(named=True):
            started = time.monotonic()
            models = monthly_models(row=row, arrays=arrays[row["site"]], combined=combined, seed=0)
            pieces.append(predict_members(row=row, arrays=arrays[row["site"]], models=models))
            print(f"monthly {row['site']}: {time.monotonic() - started:.0f} s", flush=True)
        write(arm="monthly", seed=0, pieces=pieces, smoke=args.smoke)


if __name__ == "__main__":
    main()
