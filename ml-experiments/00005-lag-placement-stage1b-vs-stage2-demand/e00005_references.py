"""Fit experiment 00005's two reference arms: R1 (ENS-trained, lags as inputs) and R2 (climatology).

R1 is one XGBoost model per series trained on the control member of every daily ENS run from
2024-04-01 to 2025-06-30, every lead, with the stage 1a input columns plus the four power lags
(null where the lag is no longer than the lead, as in experiment 00003's frames). It is applied to
each of the 51 members of the validation runs, and the quantiles are the 9 empirical quantiles of
the 51 forecasts.

R2 is the climatology by local month, half-hour of the day, and weekday or weekend, with the 9
quantiles of the training power pooled over the neighbouring months and half-hours of the same day
type (the cell definition of the `ClimatologyForecaster` in `baseline_forecasters`). It trains on
2019-09 to before 2025-06-17 and is repeated once per seed.

Output: `work/references/<r1|r2>.parquet` with the validation keys, `seed`, the 9 quantile columns,
and `actual` (power over effective capacity, null where unobserved).

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_references.py
"""

import time
from pathlib import Path
from typing import Final

import e00005_common as common
import numpy as np
import polars as pl

OUT_DIR: Final[Path] = common.WORK_DIR / "references"
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "power_lag_24h",
    "power_lag_48h",
    "power_lag_168h",
    "power_lag_336h",
)
OFFSETS: Final[tuple[tuple[int, int], ...]] = tuple(
    (month, half_hour) for month in (-1, 0, 1) for half_hour in (-1, 0, 1)
)
"""The nine cells of a climatology cell's neighbourhood."""


def _validation_cases() -> pl.DataFrame:
    """Return every validation case of the 20 series, with the actual power over capacity."""
    series = common.series_table()
    arrays = common.load_series_arrays(series=series, config="from2023")
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    keys = pl.concat(
        [
            pl.read_parquet(path, columns=["time_series_id", "power_fcst_init_time", "valid_time"])
            .filter(pl.col("time_series_id").is_in(list(ids.values())))
            .unique()
            for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
        ]
    )
    pieces = []
    for site, series_id in ids.items():
        part = keys.filter(pl.col("time_series_id") == series_id)
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        pieces.append(
            part.with_columns(actual=pl.Series(common.take(values=arrays[site].power, index=index)))
        )
    return pl.concat(pieces).sort("time_series_id", "power_fcst_init_time", "valid_time")


def r1_quantiles() -> pl.DataFrame:
    """Return R1's validation quantiles, for every series and seed.

    Returns:
        Rows keyed by series, run, valid time, and seed, with the 9 quantile columns.
    """
    series = common.series_table()
    columns = [*common.feature_columns(config="from2023"), *LAG_COLUMNS]
    train = pl.read_parquet(common.ENS_TRAIN_PATH)
    pieces = []
    for row in series.iter_rows(named=True):
        series_id, capacity = row["time_series_id"], row["effective_capacity_mw"]
        own = common.with_derived(
            frame=train.filter(
                (pl.col("time_series_id") == series_id) & pl.col("power").is_finite()
            ),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        features = own.select(columns).to_numpy().astype(np.float32)
        target = (own["power"] / capacity).to_numpy()
        models = {
            seed: common.fit_booster(features=features, target=target, seed=seed)
            for seed in common.SEEDS
        }
        for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
            run = common.with_derived(
                frame=pl.read_parquet(path).filter(pl.col("time_series_id") == series_id),
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
            ordered = run.sort("valid_time", "ensemble_member")
            members = ordered["ensemble_member"].n_unique()
            matrix = ordered.select(columns).to_numpy().astype(np.float32)
            keys = ordered.select("time_series_id", "power_fcst_init_time", "valid_time").unique(
                maintain_order=True
            )
            for seed, model in models.items():
                forecast = common.predict(model=model, features=matrix).reshape(-1, members)
                quantiles = np.quantile(forecast, common.LEVELS, axis=1).T.astype(np.float32)
                pieces.append(
                    keys.with_columns(
                        pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS), orient="row"),
                        seed=pl.lit(seed).cast(pl.Int8),
                    )
                )
    return pl.concat(pieces)


def _cells(*, valid_time: pl.Series) -> pl.DataFrame:
    local = valid_time.dt.convert_time_zone("Europe/London")
    return pl.DataFrame(
        {
            "month": local.dt.month().cast(pl.Int8),
            "half_hour": (local.dt.hour() * 2 + local.dt.minute() // 30).cast(pl.Int8),
            "weekend": local.dt.weekday() >= 6,
        }
    )


def r2_quantiles(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Return R2's quantiles for each validation case, repeated for every seed.

    Args:
        cases: `_validation_cases()`.

    Returns:
        Rows keyed by series, run, valid time, and seed, with the 9 quantile columns.
    """
    series = common.series_table()
    arrays = common.load_series_arrays(series=series, config="from2023")
    first = int(common.grid_index(times=common.GRID_START))
    stop = int(common.grid_index(times=common.FINAL_TRAIN_CUTOFF))
    pieces = []
    for row in series.iter_rows(named=True):
        data = arrays[row["site"]]
        training = pl.concat(
            [
                _cells(valid_time=data.valid_time.slice(first, stop - first)),
                pl.DataFrame({"power": data.power[first:stop]}),
            ],
            how="horizontal",
        ).filter(pl.col("power").is_finite())
        pooled = pl.concat(
            [
                training.with_columns(
                    month=((pl.col("month") - 1 + dm) % 12 + 1).cast(pl.Int8),
                    half_hour=((pl.col("half_hour") + dh) % 48).cast(pl.Int8),
                )
                for dm, dh in OFFSETS
            ]
        )
        table = pooled.group_by("month", "half_hour", "weekend").agg(
            *[
                pl.col("power").quantile(float(level)).alias(name)
                for level, name in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True)
            ]
        )
        own = cases.filter(pl.col("time_series_id") == row["time_series_id"])
        looked_up = pl.concat([own, _cells(valid_time=own["valid_time"])], how="horizontal").join(
            table, on=["month", "half_hour", "weekend"], how="left", validate="m:1"
        )
        pieces.extend(
            looked_up.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                *common.LEVEL_COLUMNS,
                seed=pl.lit(seed).cast(pl.Int8),
            )
            for seed in common.SEEDS
        )
    return pl.concat(pieces)


def main() -> None:
    """Write the reference arms' validation quantiles, skipping files that exist."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _validation_cases()
    for name in ("r1", "r2"):
        target = OUT_DIR / f"{name}.parquet"
        if target.exists():
            continue
        started = time.monotonic()
        built = r1_quantiles() if name == "r1" else r2_quantiles(cases=cases)
        frame = built.join(
            cases.select("time_series_id", "power_fcst_init_time", "valid_time", "actual"),
            on=["time_series_id", "power_fcst_init_time", "valid_time"],
            how="left",
            validate="m:1",
        )
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{name}: {frame.height:,} rows in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
