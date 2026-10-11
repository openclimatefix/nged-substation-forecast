"""Read the fold, the series, the observed power, and the ENS weather the quick-wins arms share.

The Dagster research assets read these inputs through `nged_substation_forecast.defs`, which a
study may not import. This module reads the same tables with the same filters, so an arm sees the
rows the production forecast sees.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import patito as pt
import polars as pl
import yaml
from contracts.config_schemas import CvFoldConfig, load_cv_config
from contracts.power_schemas import EffectiveCapacity, PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import PROJECT_ROOT, Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import Nwp
from ml_core.cv_helpers import date_to_utc_datetime
from nged_data.storage import scan_cleaned_power
from studies.sources import REPO_DATA_DIR
from xgboost_forecaster.forecaster import XGBoostConfig

FOLD_A: Final[str] = "mid_2025_to_mid_2026"
"""The leaderboard fold, and the fold every round of the study selects on."""

OUTPUT_ROOT: Final[Path] = REPO_DATA_DIR / "studies" / "per_study" / "quick_wins"
"""Where the study writes every table and prediction file."""

SAMPLE_STRIDE_DAYS: Final[int] = 7
"""Every arm is scored on ECMWF runs initialised this many days apart, starting at `val_start`."""

MAX_NWP_LEAD: Final[timedelta] = timedelta(days=16)
"""Upper bound on a run's horizon, used to prune the initialisation-time partitions scanned."""

PRODUCTION_MODEL_CONFIG: Final[Path] = PROJECT_ROOT / "conf" / "model" / "xgboost.yaml"
"""The production forecaster's configuration."""


class FoldWindow(NamedTuple):
    """A fold's inclusive training and validation windows in UTC."""

    fold_id: str
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime


def fold_window(*, fold_id: str = FOLD_A) -> FoldWindow:
    """Return the training and validation windows of a leaderboard fold.

    Args:
        fold_id: A fold in `conf/cv/default.yaml`.

    Returns:
        The windows, with both ends inclusive, as the Dagster research assets use them.
    """
    cv_config = load_cv_config(PROJECT_ROOT / "conf" / "cv" / "default.yaml")
    fold: CvFoldConfig = cv_config.get_fold(fold_id)
    return FoldWindow(
        fold_id=fold_id,
        train_start=date_to_utc_datetime(fold.train_start),
        train_end=date_to_utc_datetime(fold.train_end, end_of_day=True),
        val_start=date_to_utc_datetime(fold.val_start),
        val_end=date_to_utc_datetime(fold.val_end, end_of_day=True),
    )


FOLD_B: Final[str] = "backwards_2025_07_to_2026_06"
"""The robustness fold: train on the leaderboard fold's validation year, validate on the months
before its training window ended. It is not in `conf/cv/default.yaml` and never feeds the
leaderboard."""


def backwards_fold_window() -> FoldWindow:
    """Return fold B, which trains on 2025-07 to 2026-06 and validates on 2024-04 to 2025-06."""
    forward = fold_window()
    return FoldWindow(
        fold_id=FOLD_B,
        train_start=forward.val_start,
        train_end=forward.val_end,
        val_start=forward.train_start,
        val_end=forward.train_end,
    )


def window_for(*, fold_id: str) -> FoldWindow:
    """Return the windows of fold A or fold B by name."""
    return backwards_fold_window() if fold_id == FOLD_B else fold_window(fold_id=fold_id)


def sample_init_times(*, window: FoldWindow) -> list[datetime]:
    """Return the ECMWF initialisation times every arm is scored on.

    Args:
        window: The fold's windows.

    Returns:
        One 00:00 UTC run every `SAMPLE_STRIDE_DAYS` days from `val_start`, up to `val_end`.
    """
    init_times = []
    init_time = window.val_start
    while init_time <= window.val_end:
        init_times.append(init_time)
        init_time += timedelta(days=SAMPLE_STRIDE_DAYS)
    return init_times


def production_config() -> XGBoostConfig:
    """Return the production forecaster's configuration, read from `conf/model/xgboost.yaml`."""
    raw = yaml.safe_load(PRODUCTION_MODEL_CONFIG.read_text())
    return XGBoostConfig(**raw["model_params"])


def eligible_series_ids(*, settings: Settings, fold_id: str) -> list[int]:
    """Return the series the leaderboard scores for a fold."""
    return sorted(
        pl.scan_delta(
            settings.eligible_time_series_data_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        )
        .filter(pl.col("fold_id") == fold_id)
        .select("time_series_id")
        .collect()["time_series_id"]
        .to_list()
    )


def series_metadata(
    *, settings: Settings, time_series_ids: list[int]
) -> pt.DataFrame[TimeSeriesMetadata]:
    """Read the metadata table, kept to the requested series, and check every series has a row."""
    metadata = pt.DataFrame(
        pl.read_parquet(
            settings.metadata_path, storage_options=typeddict_to_dict(settings.storage_options)
        ).filter(pl.col("time_series_id").is_in(time_series_ids))
    ).set_model(TimeSeriesMetadata)
    missing = sorted(set(time_series_ids) - set(metadata["time_series_id"].to_list()))
    if missing:
        msg = f"{len(missing)} series have no metadata row: {missing[:20]}"
        raise ValueError(msg)
    return metadata


def effective_capacity(*, settings: Settings) -> pt.DataFrame[EffectiveCapacity]:
    """Read the full-history effective capacity that normalises every error."""
    return EffectiveCapacity.validate(
        pl.read_delta(
            settings.effective_capacity_data_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        )
    )


def engineering_inputs(
    *,
    settings: Settings,
    time_series_ids: list[int],
    metadata: pt.DataFrame[TimeSeriesMetadata],
    window_start: datetime,
    window_end: datetime,
    ensemble_members: list[int] | None = None,
    init_time_start: datetime | None = None,
    init_time_end: datetime | None = None,
    power_lookback: timedelta = timedelta(0),
) -> tuple[pt.LazyFrame[PowerTimeSeries], pt.LazyFrame[Nwp]]:
    """Scan the cleaned power and the ENS weather for a window, pruned at the source.

    The scans and filters are those of the production loader: initialisation time prunes whole
    partitions, the ensemble-member filter skips row groups, and only the H3 cells the series sit in
    are read. Power is read from `window_start - power_lookback`, so a power lag near the start of
    the window has history to read.

    Args:
        settings: Supplies the table paths.
        time_series_ids: The series to keep.
        metadata: The metadata of those series, whose `h3_res_5` decides which cells are read.
        window_start: Inclusive start of the valid times read.
        window_end: Inclusive end of the valid times read.
        ensemble_members: The ensemble members to keep, or every member when `None`.
        init_time_start: Earliest run read, or `window_start - MAX_NWP_LEAD` when `None`.
        init_time_end: Latest run read, or `window_end` when `None`.
        power_lookback: How far before `window_start` to read power.

    Returns:
        The lazy power frame and the lazy weather frame.
    """
    init_time_start = init_time_start or window_start - MAX_NWP_LEAD
    init_time_end = init_time_end or window_end
    power = scan_cleaned_power(
        settings.cleaned_power_time_series_data_path, settings.storage_options
    ).filter(
        pl.col("time_series_id").is_in(time_series_ids),
        pl.col("time") >= window_start - power_lookback,
        pl.col("time") <= window_end,
    )
    nwp = Nwp.scan_delta(settings.nwp_data_path, storage_options=settings.storage_options).filter(
        pl.col("init_time") >= init_time_start,
        pl.col("init_time") <= init_time_end,
        pl.col("valid_time") >= window_start,
        pl.col("valid_time") <= window_end,
        pl.col("h3_index").is_in(metadata["h3_res_5"].unique().to_list()),
    )
    if ensemble_members is not None:
        nwp = nwp.filter(pl.col("ensemble_member").is_in(ensemble_members))
    return (
        pt.LazyFrame.from_existing(power).set_model(PowerTimeSeries),
        pt.LazyFrame.from_existing(nwp).set_model(Nwp),
    )
