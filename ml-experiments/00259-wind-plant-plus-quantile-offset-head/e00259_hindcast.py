"""Build the daily ENS hindcast member rows of experiment 00259, one parquet file per run.

For every 00 UTC ensemble run from 2024-04-01 to 2025-06-16 the production feature engineer
(`ml_core`'s `TabularFeatureEngineer`, as the production service and the quick-wins study call it)
turns the ENS table's rows at the three farms' H3 cells into half-hourly member rows. The columns
equal those of `e00259_common.ens_member_rows`, which reproduces the 53 Tuesday runs of the scored
year, so the head trains on the same inputs it is scored on. Valid times at or after `TRAIN_END`
are dropped.

Output: `work/hindcast/init=<date>.parquet` and `work/hindcast/checks.json`.

    uv run --no-sync python ml-experiments/00259-*/e00259_hindcast.py
"""

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Final

import e00259_common as common
import numpy as np
import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import Nwp
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer
from nged_data.storage import scan_cleaned_power
from studies.sources import REPO_DATA_DIR

HINDCAST_DIR: Final = common.WORK_DIR / "hindcast"
FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2025, 6, 16, tzinfo=UTC)
MAX_LEAD: Final[timedelta] = timedelta(days=15)
ENGINEERED: Final[frozenset[str]] = frozenset(
    {
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "temperature_2m",
        "nwp_lead_time_hours",
    }
)
MIN_COMPLETE_SHARE: Final[float] = 0.95


def settings() -> Settings:
    """Return the settings with every data path under the shared data folder."""
    root = str(REPO_DATA_DIR)
    return Settings(data_path_internal=root, data_path_delivery=root, local_artifacts_path=root)


def build_run(
    *, settings: Settings, metadata: pt.DataFrame[TimeSeriesMetadata], init: datetime
) -> pl.DataFrame:
    """Return one run's member rows at the farms: the key columns, member, and `WEATHER_COLUMNS`."""
    ids = metadata["time_series_id"].to_list()
    end = min(init + MAX_LEAD, common.TRAIN_END - timedelta(minutes=30))
    power = scan_cleaned_power(
        settings.cleaned_power_time_series_data_path, settings.storage_options
    ).filter(pl.col("time_series_id").is_in(ids), pl.col("time") >= init, pl.col("time") <= end)
    nwp = Nwp.scan_delta(settings.nwp_data_path, storage_options=settings.storage_options).filter(
        pl.col("init_time") == init,
        pl.col("valid_time") >= init,
        pl.col("valid_time") <= end,
        pl.col("h3_index").is_in(metadata["h3_res_5"].unique().to_list()),
    )
    frame = pl.DataFrame(
        TabularFeatureEngineer()
        .engineer(
            selected_features=set(ENGINEERED),
            power_time_series=pt.LazyFrame.from_existing(power).set_model(PowerTimeSeries),
            time_series_metadata=metadata,
            nwp=pt.LazyFrame.from_existing(nwp).set_model(Nwp),
        )
        .collect(engine="streaming")
    )
    if frame.height == 0:
        return frame
    bearing_10 = pl.col("wind_direction_10m") * np.pi / 180.0
    bearing_100 = pl.col("wind_direction_100m") * np.pi / 180.0
    return (
        frame.select(
            *common.KEY_COLUMNS,
            "ensemble_member",
            "wind_speed_10m",
            "wind_speed_100m",
            bearing_10.sin().alias("wind_dir_sin_10m"),
            bearing_10.cos().alias("wind_dir_cos_10m"),
            bearing_100.sin().alias("wind_dir_sin_100m"),
            bearing_100.cos().alias("wind_dir_cos_100m"),
            pl.col("temperature_2m").alias("temperature_c"),
        )
        .with_columns(pl.col(common.WEATHER_COLUMNS).cast(pl.Float32))
        .filter(pl.col("valid_time") < common.TRAIN_END)
        .sort(*common.KEY_COLUMNS, "ensemble_member")
    )


def main() -> None:
    """Write every run's member rows, skipping finished runs, and the completeness checks."""
    HINDCAST_DIR.mkdir(parents=True, exist_ok=True)
    config = settings()
    sites = common.site_table()
    ids = sites["time_series_id"].to_list()
    metadata = pt.DataFrame(
        pl.read_parquet(
            config.metadata_path, storage_options=typeddict_to_dict(config.storage_options)
        ).filter(pl.col("time_series_id").is_in(ids))
    ).set_model(TimeSeriesMetadata)
    checks: dict[str, dict] = {}
    init = FIRST_RUN
    while init <= LAST_RUN:
        path = HINDCAST_DIR / f"init={init:%Y-%m-%d}.parquet"
        if not path.exists():
            frame = build_run(settings=config, metadata=metadata, init=init)
            if frame.height:
                frame.write_parquet(path.with_suffix(".tmp"))
                path.with_suffix(".tmp").rename(path)
            checks[f"{init:%Y-%m-%d}"] = {"rows": frame.height}
            print(f"{init:%Y-%m-%d}: {frame.height:,} rows", flush=True)
        init += timedelta(days=1)
    (HINDCAST_DIR / "checks.json").write_text(json.dumps(checks, indent=1))
    total_days = (LAST_RUN - FIRST_RUN).days + 1
    present = len(list(HINDCAST_DIR.glob("init=*.parquet")))
    print(f"{present} of {total_days} runs present ({present / total_days:.1%})")
    if present / total_days < MIN_COMPLETE_SHARE:
        msg = "fewer than 95% of the daily runs have rows at the farms' cells"
        raise SystemExit(msg)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
