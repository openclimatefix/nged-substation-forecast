"""Pure, data-only helpers for the cross-validation Dagster assets.

Every function here is deliberately free of I/O (no Delta, MLflow, or Dagster imports) so it
can be unit-tested in isolation. The CV asset bodies stay thin by delegating their logic here.
"""

import calendar
from datetime import date, datetime, time, timezone
from typing import Any, Final

import polars as pl
from contracts.hydra_schemas import CvFoldConfig
from pydantic import BaseModel

CV_PARTITION_KEY_SEPARATOR: Final[str] = "__"
"""Separator between experiment name and fold id in a CV partition key.

A double-underscore reduces collision risk with experiment names that contain single underscores.
"""


def _date_to_utc_datetime(d: date, *, end_of_day: bool = False) -> datetime:
    """Return a tz-aware UTC datetime at the start (or inclusive end) of the given date.

    Args:
        d: The calendar date.
        end_of_day: If True, return ``d`` at ``23:59:59`` (the inclusive end-of-day used by
            both the training window and ``val_end``); otherwise ``00:00:00``.
    """
    clock = time(23, 59, 59) if end_of_day else time(0, 0, 0)
    return datetime.combine(d, clock, tzinfo=timezone.utc)


def _subtract_months(d: date, months: int) -> date:
    """Return the date ``months`` calendar months before ``d``, clamping the day-of-month.

    Clamping handles the case where the source day does not exist in the target month
    (e.g. subtracting one month from 31 March yields 28/29 February).
    """
    month_index = d.year * 12 + (d.month - 1) - months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def training_window(fold: CvFoldConfig) -> tuple[datetime, datetime]:
    """Return the ``[start, end]`` training window for a fold, in UTC.

    Uses **inclusive end-of-day** semantics that mirror ``val_end``: the window is
    ``[train_start 00:00:00, train_end 23:59:59]``. When a fold leaves a gap/embargo between
    ``train_end`` and ``val_start``, training correctly stops at ``train_end``, not at
    ``val_start``.
    """
    return (
        _date_to_utc_datetime(fold.train_start),
        _date_to_utc_datetime(fold.train_end, end_of_day=True),
    )


def eligible_time_series_ids(
    coverage: pl.DataFrame,
    fold: CvFoldConfig,
    min_training_months: int,
) -> list[int]:
    """Return the sorted ``time_series_id``s eligible for a fold, from data coverage alone.

    A time series is eligible when it has at least ``min_training_months`` of observations
    before the fold's ``val_start`` **and** observations through the fold's ``val_end``.

    Eligibility is a function of the **data only** — it does not depend on any model or
    experiment config — so every experiment evaluates a fold on the identical population,
    which is what makes leaderboard comparisons fair.

    Args:
        coverage: One row per time series with the columns ``time_series_id``, ``first_time``,
            and ``last_time`` (the min and max observation timestamps, tz-aware UTC).
        fold: The CV fold whose eligibility is being computed.
        min_training_months: Minimum months of pre-``val_start`` history required.

    Returns:
        Sorted list of eligible ``time_series_id`` values.
    """
    earliest_required_first_time = _date_to_utc_datetime(
        _subtract_months(fold.val_start, min_training_months)
    )
    val_end_dt = _date_to_utc_datetime(fold.val_end, end_of_day=True)
    eligible = coverage.filter(
        (pl.col("first_time") <= earliest_required_first_time) & (pl.col("last_time") >= val_end_dt)
    )
    return sorted(eligible["time_series_id"].to_list())


def parse_cv_partition_key(partition_key: str) -> tuple[str, str]:
    """Return ``(experiment_name, fold_id)`` from a CV partition key.

    Partition key format: ``"{experiment_name}__{fold_id}"``. The separator is a
    double-underscore to reduce collision risk with experiment names that contain single
    underscores. Splitting from the right keeps ``__`` inside the experiment name intact.
    """
    experiment_name, fold_id = partition_key.rsplit(CV_PARTITION_KEY_SEPARATOR, maxsplit=1)
    return experiment_name, fold_id


def flatten_config(config: BaseModel | dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a (possibly nested) config into dotted-key string params for MLflow.

    MLflow params are scalar strings, so nested dicts are flattened with dotted keys and every
    leaf value is stringified. A pydantic ``BaseModel`` is first dumped to a plain dict.

    Args:
        config: A pydantic model or (possibly nested) dict to flatten.
        prefix: Internal recursion prefix; callers should not set it.

    Returns:
        A flat ``{dotted_key: str_value}`` mapping suitable for ``mlflow.log_params``.
    """
    if isinstance(config, BaseModel):
        config = config.model_dump(mode="json")
    flat: dict[str, str] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=f"{full_key}."))
        else:
            flat[full_key] = str(value)
    return flat
