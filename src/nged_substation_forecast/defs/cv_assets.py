"""Cross-validation Dagster assets.

These assets implement the experiment-independent, fold-partitioned CV layer. Each asset is a
thin orchestration shell delegating its logic to the pure helpers in ``ml_core._cv_helpers`` so
the logic stays fast to unit-test and the assets stay readable.
"""

import polars as pl
from contracts.hydra_schemas import load_cv_config
from contracts.ml_schemas import EligibleTimeSeries
from contracts.settings import Settings
from dagster import AssetExecutionContext, StaticPartitionsDefinition, asset
from deltalake import write_deltalake
from ml_core._cv_helpers import eligible_time_series_ids

# The CV folds are the shared leaderboard evaluation protocol, read from conf/cv/default.yaml
# (never hard-coded) so every experiment and asset agrees on the same folds. Loaded at import so
# the partition keys are available when Dagster builds the asset graph. PROJECT_ROOT is used here
# (rather than Settings, which needs the .env secrets) so the partition set can be built without
# any credentials.
_cv_config = load_cv_config(Settings.model_fields["cv_config_path"].default)

cv_fold_partitions = StaticPartitionsDefinition(_cv_config.fold_ids)
"""One partition per canonical CV fold (by fold year, e.g. "2022"). Experiment-independent."""


@asset(
    partitions_def=cv_fold_partitions,
    deps=["power_time_series_and_metadata"],
)
def eligible_time_series(context: AssetExecutionContext) -> None:
    """Compute and persist the canonical eligible ``time_series_id``s for one CV fold.

    A time series is eligible for a fold when its observed-power coverage has at least
    ``min_training_months`` of history before the fold's ``val_start`` *and* reaches the fold's
    ``val_end``. Eligibility is derived from data coverage alone (not from any model/config), so
    every experiment evaluates the fold on the identical population — this is what keeps the
    leaderboard apples-to-apples (see plan §4.5.1).

    The result is written to the ``eligible_time_series`` Delta table as one partition per
    ``fold_id`` via an idempotent partition overwrite, so re-materialising a fold replaces its
    rows rather than duplicating them.
    """
    settings = Settings()
    fold_id = context.partition_key
    fold = _cv_config.get_fold(fold_id)

    power_path = settings.nged_data_path / "power_time_series.delta"
    coverage = (
        pl.scan_delta(str(power_path))
        .group_by("time_series_id")
        .agg(
            first_time=pl.col("time").min(),
            last_time=pl.col("time").max(),
        )
        .collect()
    )

    eligible_ids = eligible_time_series_ids(
        coverage, fold, min_training_months=_cv_config.min_training_months
    )

    eligible_df = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([fold_id] * len(eligible_ids), dtype=pl.String),
                "time_series_id": pl.Series(eligible_ids, dtype=pl.Int32),
            }
        )
    )

    settings.eligible_time_series_data_path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        table_or_uri=settings.eligible_time_series_data_path,
        data=eligible_df.to_arrow(),
        mode="overwrite",
        predicate=f"fold_id = '{fold_id}'",
        partition_by=["fold_id"],
    )

    context.add_output_metadata(
        {
            "fold_id": fold_id,
            "n_eligible_time_series": len(eligible_ids),
            "n_time_series_in_coverage": len(coverage),
            "eligible_time_series_ids": str(eligible_ids),
            "val_start": str(fold.val_start),
            "val_end": str(fold.val_end),
            "min_training_months": _cv_config.min_training_months,
        }
    )
