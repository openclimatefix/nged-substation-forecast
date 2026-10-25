"""Build the daily ENS member rows at the three farms for the runs after experiment 00259.

Same feature engineering as `e00259_hindcast.build_run`, without the cap at 2025-06-17 on valid
times: one parquet file per 00 UTC run from 2025-06-17 to 2026-10-23, valid times up to 15 days
after the run. Output: `work/members/init=<date>.parquet`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      env PYTHONPATH=<see e00346_common> uv run --no-sync python e00346_members.py [--workers 6]
"""

import argparse
import multiprocessing
import os
import time
from datetime import timedelta

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def build_one(init_text: str) -> tuple[str, int, float]:
    """Write one run's member rows; return its date, row count, and seconds."""
    from datetime import datetime

    import e00259_common as c259  # ty: ignore[unresolved-import]
    import e00259_hindcast as hind  # ty: ignore[unresolved-import]
    import e00346_common as common
    import numpy as np
    import patito as pt
    import polars as pl
    from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
    from contracts.typing_utils import typeddict_to_dict
    from contracts.weather_schemas import Nwp
    from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer
    from nged_data.storage import scan_cleaned_power

    started = time.time()
    init = datetime.fromisoformat(init_text)
    path = common.MEMBERS_DIR / f"init={init:%Y-%m-%d}.parquet"
    if path.exists():
        return init_text, -1, 0.0
    config = hind.settings()
    sites = c259.site_table()
    ids = sites["time_series_id"].to_list()
    metadata = pt.DataFrame(
        pl.read_parquet(
            config.metadata_path, storage_options=typeddict_to_dict(config.storage_options)
        ).filter(pl.col("time_series_id").is_in(ids))
    ).set_model(TimeSeriesMetadata)
    end = init + hind.MAX_LEAD
    power = scan_cleaned_power(
        config.cleaned_power_time_series_data_path, config.storage_options
    ).filter(pl.col("time_series_id").is_in(ids), pl.col("time") >= init, pl.col("time") <= end)
    nwp = Nwp.scan_delta(config.nwp_data_path, storage_options=config.storage_options).filter(
        pl.col("init_time") == init,
        pl.col("valid_time") >= init,
        pl.col("valid_time") <= end,
        pl.col("h3_index").is_in(metadata["h3_res_5"].unique().to_list()),
    )
    frame = pl.DataFrame(
        TabularFeatureEngineer()
        .engineer(
            selected_features=set(hind.ENGINEERED),
            power_time_series=pt.LazyFrame.from_existing(power).set_model(PowerTimeSeries),
            time_series_metadata=metadata,
            nwp=pt.LazyFrame.from_existing(nwp).set_model(Nwp),
        )
        .collect(engine="streaming")
    )
    if frame.height:
        bearing_10 = pl.col("wind_direction_10m") * np.pi / 180.0
        bearing_100 = pl.col("wind_direction_100m") * np.pi / 180.0
        frame = (
            frame.select(
                *c259.KEY_COLUMNS,
                "ensemble_member",
                "wind_speed_10m",
                "wind_speed_100m",
                bearing_10.sin().alias("wind_dir_sin_10m"),
                bearing_10.cos().alias("wind_dir_cos_10m"),
                bearing_100.sin().alias("wind_dir_sin_100m"),
                bearing_100.cos().alias("wind_dir_cos_100m"),
                pl.col("temperature_2m").alias("temperature_c"),
            )
            .with_columns(pl.col(c259.WEATHER_COLUMNS).cast(pl.Float32))
            .sort(*c259.KEY_COLUMNS, "ensemble_member")
        )
        frame.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
    return init_text, frame.height, time.time() - started


def main() -> None:
    """Build every missing run with a spawn pool."""
    import e00346_common as common

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    common.MEMBERS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    day = common.FIRST_NEW_RUN
    while day <= common.LAST_NEW_RUN:
        runs.append(day.isoformat())
        day += timedelta(days=1)
    if args.limit:
        runs = runs[: args.limit]
    context = multiprocessing.get_context("spawn")
    done = 0
    with context.Pool(args.workers) as pool:
        for text, rows, seconds in pool.imap_unordered(build_one, runs):
            done += 1
            print(f"{done}/{len(runs)} {text[:10]}: {rows:,} rows, {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
