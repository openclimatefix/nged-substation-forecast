# ruff: noqa: E501
"""Build the head's input table: the stored best guess, the row mask, and the ENS wind summary.

Joins, for every wind case of experiment 33's best guess, (1) whether experiment 210's mask drops the
row, (2) the ensemble mean and standard deviation of the 100 m wind speed and the direction sector
of the ensemble-mean wind, and (3) the ERA5-trained wind model's nine quantiles when it is fed the
ensemble-mean wind (the base forecast of arm A4; the speed is the mean of the members' speeds and
only the direction comes from the vector-mean wind), refitted per seed as in experiment 52.

Output: `work/inputs.parquet` (one row per case and seed). Existing outputs are kept.

    uv run --no-sync python ml-experiments/00221-wind-correction-head-on-stored-predictions/e00221_inputs.py
"""

import json
import time
from pathlib import Path
from typing import Final

import e00221_common as common
import numpy as np
import polars as pl
from e00221_common import KEY_COLUMNS, LEVEL_COLUMNS, WIND_FEATURES
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ENS_DIR: Final[Path] = common.PILOT_INPUTS / "ens_features" / "val"
MASK_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00210-wind-idle-noise-mva-artefact" / "mask.parquet"
)
OUT_PATH: Final[Path] = common.WORK_DIR / "inputs.parquet"
MEAN_ONLY_PREFIX: Final[str] = "mo_"
N_SECTORS: Final[int] = 8
HEIGHTS: Final[tuple[str, ...]] = ("10", "100")


def ens_summary(*, series: list[int]) -> pl.DataFrame:
    """Return, per case, the 100 m speed mean and spread and the ensemble-mean wind features."""
    pieces = []
    for path in sorted(ENS_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(series))
        columns = []
        for h in HEIGHTS:
            speed = pl.col(f"wind_speed_{h}m").cast(pl.Float64)
            columns += [
                (-speed * pl.col(f"wind_dir_sin_{h}m")).mean().alias(f"u{h}"),
                (-speed * pl.col(f"wind_dir_cos_{h}m")).mean().alias(f"v{h}"),
                speed.mean().alias(f"scalar{h}"),
            ]
        columns += [
            pl.col("wind_speed_100m").cast(pl.Float64).mean().alias("ens_speed_mean"),
            pl.col("wind_speed_100m").cast(pl.Float64).std().alias("ens_speed_std"),
            pl.col("ensemble_member").n_unique().alias("members"),
        ]
        pieces.append(run.group_by(*KEY_COLUMNS).agg(columns))
    out = pl.concat(pieces)
    if (out["members"] != 51).any():
        msg = "a case lacks some of the 51 members"
        raise ValueError(msg)
    features = []
    for h in HEIGHTS:
        magnitude = (pl.col(f"u{h}") ** 2 + pl.col(f"v{h}") ** 2).sqrt()
        features += [
            pl.col(f"scalar{h}").alias(f"mo_wind_speed_{h}m"),
            (-pl.col(f"u{h}") / magnitude.clip(lower_bound=1e-6)).alias(f"mo_wind_dir_sin_{h}m"),
            (-pl.col(f"v{h}") / magnitude.clip(lower_bound=1e-6)).alias(f"mo_wind_dir_cos_{h}m"),
        ]
    out = out.with_columns(features)
    bearing = pl.arctan2(pl.col("mo_wind_dir_sin_100m"), pl.col("mo_wind_dir_cos_100m"))
    sector = ((bearing % (2 * np.pi)) / (2 * np.pi) * N_SECTORS + 0.5).floor() % N_SECTORS
    return out.with_columns(sector=sector.cast(pl.Int32)).drop(
        "members", *[f"{c}{h}" for c in ("u", "v", "scalar") for h in HEIGHTS]
    )


def mean_only_quantiles(*, summary: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the wind model's nine quantiles at the ensemble-mean wind, per seed and case."""
    analysis = pl.read_parquet(common.ANALYSIS_PATH)
    observed = common.observed_training_power(sites=sites)
    frame = summary.select(
        *KEY_COLUMNS,
        **{f: pl.col(f"{MEAN_ONLY_PREFIX}{f}") for f in WIND_FEATURES},
    )
    pieces = []
    for seed in common.SEEDS:
        started = time.monotonic()
        for series in sites["time_series_id"].to_list():
            training = (
                analysis.filter(pl.col("time_series_id") == series)
                .join(
                    observed.filter(pl.col("time_series_id") == series),
                    on=["time_series_id", "valid_time"],
                    how="inner",
                    validate="1:1",
                )
                .filter(
                    (pl.col("valid_time") >= common.ERA5_START)
                    & (pl.col("valid_time") < common.TRAIN_END)
                )
            )
            model = common.WindModel(rows=training, seed=seed, rounds=common.N_ROUNDS)
            part = frame.filter(pl.col("time_series_id") == series)
            quantiles = model.predict(frame=part)
            pieces.append(
                part.select(*KEY_COLUMNS, seed=pl.lit(seed, dtype=pl.Int32)).hstack(
                    pl.DataFrame(
                        quantiles.astype(np.float32),
                        schema=[f"{MEAN_ONLY_PREFIX}{c}" for c in LEVEL_COLUMNS],
                    )
                )
            )
        print(f"seed {seed} refit in {time.monotonic() - started:.0f} s", flush=True)
    return pl.concat(pieces)


def main() -> None:
    """Write `work/inputs.parquet` and a short check file."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"{OUT_PATH} exists; keeping it", flush=True)
        return
    sites = common.site_table()
    series = sites["time_series_id"].to_list()
    cases = (
        pl.read_parquet(common.BEST_GUESS_PATH)
        .filter(pl.col("technology") == "wind")
        .with_columns(
            lead_day=common.lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"),
                valid_time=pl.col("valid_time"),
            )
        )
        .drop("technology", "band")
    )
    mask = pl.read_parquet(MASK_PATH).select(
        "time_series_id", "valid_time", masked=pl.lit(value=True), drop_reason=pl.col("drop_reason")
    )
    summary = ens_summary(series=series)
    mean_only = mean_only_quantiles(summary=summary, sites=sites)
    out = (
        cases.join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(pl.col("masked").fill_null(value=False))
        .join(summary, on=list(KEY_COLUMNS), how="left", validate="m:1")
        .join(mean_only, on=[*KEY_COLUMNS, "seed"], how="left", validate="1:1")
        .join(sites.select("time_series_id", "site"), on="time_series_id", validate="m:1")
    )
    checks = {
        "cases": out.height,
        "rows_without_ens_summary": int(out["ens_speed_mean"].is_null().sum()),
        "rows_without_mean_only": int(out["mo_q50"].is_null().sum()),
        "masked_share_by_site": {
            site: float(share)
            for site, share in out.group_by("site")
            .agg(pl.col("masked").mean())
            .sort("site")
            .iter_rows()
        },
        "mean_speed_100m_ens_mean": float(out["ens_speed_mean"].to_numpy().mean()),
    }
    print(json.dumps(checks, indent=1), flush=True)
    if checks["rows_without_ens_summary"] or checks["rows_without_mean_only"]:
        msg = "some cases lack the ENS summary or the mean-only forecast"
        raise ValueError(msg)
    out.write_parquet(OUT_PATH.with_suffix(".tmp"))
    OUT_PATH.with_suffix(".tmp").rename(OUT_PATH)
    (common.WORK_DIR / "input_checks.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
