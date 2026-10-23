"""Leave-one-year-out fits of the core and XGBoost with ERA5 as the weather, for D3's year table.

For each July-to-June block of the training period (block 0 is September 2019 to June 2020, block 5
is July 2024 to 16 June 2025), fits the core (P) and XGBoost (X) on the other blocks of the same
farm and scores every half-hour of the held-out block that has an observation. The scored rows are
every half-hour, not experiment 33's 53 Tuesday runs, so this table is labelled as a different row
set. XGBoost uses seed 0 only.

Output: `work/loyo.parquet`, one row per farm and half-hour with both models' quantile scores.

    cd ml-experiments/00276-wind-xgboost-vs-physics-gap-diagnostics
    uv run --no-sync python e00276_loyo.py [--tag TAG NAME ...]

With no arguments it fits `P` and `X` and writes `work/loyo.parquet`. With `--tag loyo_year X5 X6`
it fits the named variants and writes `work/loyo_year.parquet`.
"""

import argparse
import multiprocessing
import os
import sys

os.environ.setdefault("E00276_TORCH_THREADS", "2")

import e00276_common as common
import e00276_models as models
import numpy as np
import polars as pl

BLOCKS = range(models.LAST_TRAIN_BLOCK + 1)


def worker(*, farm: str, names: list[str], tag: str) -> None:
    """Fit both models for every held-out block of one farm and store the scored rows."""
    out_dir = common.WORK_DIR / ("loyo" if tag == "loyo" else f"loyo_{tag}")
    path = out_dir / f"{farm}.parquet"
    if path.exists():
        return
    specs = models.build_specs()
    sites = common.site_table().filter(pl.col("site") == farm)
    era5 = common.era5_rows(sites=sites)
    table = models.add_derived_columns(frame=common.training_table(sites=sites, era5=era5))
    mask = common.idle_mask().with_columns(masked=pl.lit(value=True))
    table = table.join(mask, on=["time_series_id", "valid_time"], how="left").with_columns(
        masked=pl.col("masked").fill_null(value=False)
    )
    pieces = []
    for block in BLOCKS:
        held = table.filter(pl.col("block") == block)
        train = table.filter(pl.col("block") != block)
        scores = {}
        for name in names:
            predicted, _ = models.fit_predict(
                spec=specs[name],
                train=train,
                evals={"held": held},
                seed=0,
                with_idle_layer=farm != "W2",
            )
            scores[name] = (
                common.quantile_score(
                    quantiles=predicted["held"].astype(np.float64),
                    actual=held["target"].to_numpy().astype(np.float64),
                )
                * common.POINTS_PER_FRACTION
            )
        pieces.append(
            held.select(
                site=pl.lit(farm),
                valid_time=pl.col("valid_time"),
                block=pl.col("block"),
                speed_100m=pl.col("wind_speed_100m"),
                direction_deg=(
                    pl.arctan2(pl.col("wind_dir_sin_100m"), pl.col("wind_dir_cos_100m")).degrees()
                    % 360
                ),
                valid_hour=pl.col("hour").cast(pl.Int32),
                valid_month=pl.col("valid_time").dt.month(),
                masked=pl.col("masked"),
                **{f"qs_{name}": pl.Series(scores[name]) for name in names},
            )
        )
        print(f"{farm} block {block} done", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(pieces).write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)


def main() -> None:
    """Run one worker per farm, then join their tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="loyo")
    parser.add_argument("names", nargs="*", default=["P", "X"])
    args = parser.parse_args()
    out_dir = common.WORK_DIR / ("loyo" if args.tag == "loyo" else f"loyo_{args.tag}")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, kwargs={"farm": farm, "names": args.names, "tag": args.tag})
        for farm in ("W1", "W2", "W3")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        sys.exit("a worker failed")
    pl.concat(
        [pl.read_parquet(out_dir / f"{f}.parquet") for f in ("W1", "W2", "W3")]
    ).write_parquet(common.WORK_DIR / f"{args.tag}.parquet")


if __name__ == "__main__":
    main()
