"""Score a study's predictions file through the `metrics` asset.

A study's leaderboard number comes only from this script. The script takes a predictions file and
nothing else: no code, no actuals, no scoring options. It runs `metrics` in leaderboard scope on the
file's rows, stored in `power_forecasts` under the experiment name `study/<study name>`.

Before anything is written, the script refuses a file whose row keys `(time_series_id,
power_fcst_init_time, valid_time, ensemble_member)` differ from those of the cross-validation (CV)
config's `reference_experiment_name` experiment for the same fold. The script also refuses a file
whose rows carry more than one `power_fcst_model_name`. A study therefore cannot abstain on hard
rows, choose its own ensemble members, or down-weight rows by spreading them across model names.
The `metrics` asset repeats both checks when it scores a study, because the asset is the one source
of a leaderboard number.

The script also discards the environment it was started with. `Settings` reads `DATA_PATH_INTERNAL`,
`METADATA_PATH`, `CV_CONFIG_PATH`, and `MLFLOW_TRACKING_URI` from the environment, and the `metrics`
asset reads `NGED_FINAL_TEST`. Any of them could repoint the actuals or switch off the final-test
date guard. The script therefore re-executes itself with an environment holding only
`ALLOWED_ENVIRONMENT`. The paths and credentials come from the `.env` file of the checkout.

The script runs the code of whichever checkout its interpreter belongs to. An autonomous research
session can therefore change what the script executes, unless the maintainer's `sudo` rule runs
the `main` checkout's interpreter directly (not `uv run`) on the `main` checkout's copy of the
script, with `env_reset` and a fixed `secure_path`. The re-execution in `main` happens only after
the module's imports have run, so a `PYTHONPATH` or `PATH` set before the script starts can still
run code in the first process.

    uv run python scripts/forecasting/score_study.py \
        predictions.parquet my_study mid_2025_to_mid_2026

Every attempt is appended to `study_submissions.jsonl` in `local_artifacts_path`, whether the
script scores the file or refuses it. The log is append-only. Each scored study's MLflow fold run
carries the `study_submission_number` tag, which counts the log's attempts at that fold by every
study, including this one. Each new study name is another attempt against the same fold and the
same reference, so the number says how many chances the submitters have had to find a lucky score
on that fold.

A study that has already been scored is not overwritten. `--replace` overwrites the earlier
submission, its metrics, and its MLflow fold run, so the script accepts the flag only when
`NGED_ALLOW_REPLACE=1` is set in the environment of the maintainer's own shell. The `sudo` rule's
`env_reset` drops that variable. The log keeps the record of an overwritten submission.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import mlflow
import patito as pt
import polars as pl
from contracts.config_schemas import STUDY_EXPERIMENT_PREFIX, load_cv_config
from contracts.power_schemas import PowerForecast
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from dagster import DagsterInstance, RunConfig, materialize
from delta_store.power_forecasts import write_power_forecasts
from ml_core.metrics import ROW_KEY_COLUMNS, require_same_row_keys, require_single_model_name
from ml_core.mlflow_runs import (
    get_or_create_experiment,
    get_or_create_fold_run,
    get_or_create_parent_run,
)
from mlflow.tracking import MlflowClient

from nged_substation_forecast.defs.cv_assets import MetricsConfig, PopulationFilter, metrics

STUDY_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_-]{1,64}$")
"""A study name is lower-case letters, digits, underscores, and hyphens.

The Delta writers build their overwrite predicate from the experiment name by string formatting, so
a name containing a quote could overwrite other experiments' rows.
"""

ALLOWED_ENVIRONMENT: Final[frozenset[str]] = frozenset({"PATH", "HOME", "LANG", "LC_ALL"})
"""The environment variables the script keeps."""

CLEAN_ENVIRONMENT_MARKER: Final[str] = "NGED_SCORE_STUDY_CLEAN_ENVIRONMENT"
"""Set to `1` in the re-executed process, so the process re-executes once only."""

ALLOW_REPLACE_VARIABLE: Final[str] = "NGED_ALLOW_REPLACE"
"""The environment variable that lets `--replace` overwrite a submission.

Only the maintainer's own shell sets it to `1`. The script forwards it across the re-execution,
whose cleaned environment would otherwise drop it.
"""

SUBMISSION_LOG_NAME: Final[str] = "study_submissions.jsonl"
"""The append-only log of every submission attempt, in `local_artifacts_path`."""

SUBMISSION_NUMBER_TAG: Final[str] = "study_submission_number"
"""The MLflow fold-run tag holding the attempt's number among the log's attempts at the fold."""

SERIES_BATCH_SIZE: Final[int] = 4
"""How many `time_series_id` values to check and write at once.

A full fold is about 364 million rows, which already exhausts a 29 GB machine when written in one
call. The size matches the scoring batch of the `metrics` asset.
"""


def clean_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return only the variables of `environment` that the script keeps.

    Args:
        environment: The caller's environment.

    Returns:
        The variables named in `ALLOWED_ENVIRONMENT`.
    """
    return {name: value for name, value in environment.items() if name in ALLOWED_ENVIRONMENT}


def validate_study_name(study_name: str) -> str:
    """Return `study_name` if it matches `STUDY_NAME_PATTERN`.

    Args:
        study_name: The name the caller chose for the study.

    Returns:
        `study_name`, unchanged.

    Raises:
        ValueError: If the name does not match `STUDY_NAME_PATTERN`: an empty name, a name over 64
            characters, or a name with a character outside the pattern, such as the `/` of a
            leading `study/`.
    """
    if STUDY_NAME_PATTERN.fullmatch(study_name) is None:
        raise ValueError(
            f"Study name {study_name!r} must match {STUDY_NAME_PATTERN.pattern}: lower-case "
            "letters, digits, underscores, and hyphens, up to 64 characters, with no prefix."
        )
    return study_name


def _open_predictions(*, predictions: Path, experiment_name: str, fold_id: str) -> pl.LazyFrame:
    """Scan the predictions file, checking its row-key dtypes and fold id, and stamping both ids.

    Args:
        predictions: Parquet file of `PowerForecast` rows.
        experiment_name: The `study/<name>` experiment the rows are stored under.
        fold_id: The leaderboard fold the rows forecast.

    Returns:
        A lazy scan whose `experiment_name` column is `experiment_name` and whose `fold_id` column
        is `fold_id`.

    Raises:
        ValueError: If the file lacks a row-key column, a row-key column has a different dtype
            from `PowerForecast`'s, or a `fold_id` column holds any value except `fold_id`.
    """
    scan = pl.scan_parquet(predictions)
    schema = scan.collect_schema()
    for column in ROW_KEY_COLUMNS:
        expected = PowerForecast.dtypes[column]
        if column not in schema or schema[column] != expected:
            raise ValueError(
                f"Column {column!r} must have dtype {expected}; the file has {schema.get(column)}."
            )
    if "fold_id" in scan.collect_schema().names():
        fold_ids = scan.select("fold_id").unique().collect(engine="streaming")["fold_id"].to_list()
        if fold_ids != [fold_id]:
            raise ValueError(f"The file's fold_id values are {fold_ids}, not [{fold_id!r}].")
    return scan.with_columns(experiment_name=pl.lit(experiment_name), fold_id=pl.lit(fold_id))


def _partition_has_rows(*, forecasts: pl.LazyFrame, experiment_name: str, fold_id: str) -> bool:
    """Return whether `power_forecasts` already holds rows for the experiment and fold."""
    existing = PopulationFilter(experiment_name=experiment_name, fold_id=fold_id).apply(forecasts)
    return existing.head(1).collect(engine="streaming").height > 0


def _validated_batches(study: pl.LazyFrame) -> Iterator[pt.DataFrame[PowerForecast]]:
    """Yield the study's rows one batch of series at a time, each validated as `PowerForecast`."""
    series_ids = sorted(
        study.select("time_series_id").unique().collect(engine="streaming")["time_series_id"]
    )
    for start in range(0, len(series_ids), SERIES_BATCH_SIZE):
        batch_ids = series_ids[start : start + SERIES_BATCH_SIZE]
        yield PowerForecast.validate(
            study.filter(pl.col("time_series_id").is_in(batch_ids)).collect(engine="streaming")
        )


def _write_in_batches(
    *, study: pl.LazyFrame, settings: Settings, experiment_name: str, fold_id: str
) -> None:
    """Write the study's rows to `power_forecasts` one batch of series at a time.

    A first pass validates every batch, so a malformed row leaves no partial partition behind. The
    first batch then replaces the `(experiment_name, fold_id)` partition and the rest append to it,
    as `cv_power_forecasts` does, so peak memory is one batch.
    """
    for _ in _validated_batches(study):
        pass
    for index, batch in enumerate(_validated_batches(study)):
        write_power_forecasts(
            forecasts=batch,
            table_uri=settings.power_forecasts_data_path,
            replace_partition=(experiment_name, fold_id) if index == 0 else None,
            storage_options=settings.storage_options,
        )


def _append_to_log(*, log_path: Path, entry: dict[str, object]) -> int:
    """Append one JSON line to the submission log under an exclusive lock.

    Args:
        log_path: The submission log, created with its parent directory if absent.
        entry: The line to append. An `attempt` entry carries the fold it attempts in `fold_id`.

    Returns:
        How many `attempt` entries for the entry's fold the log now holds, including this entry.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a+", encoding="utf-8") as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.seek(0)
        earlier_attempts = 0
        for line in log:
            earlier = json.loads(line)
            is_attempt_at_fold = (
                earlier["event"] == "attempt" and earlier["fold_id"] == entry["fold_id"]
            )
            earlier_attempts += is_attempt_at_fold
        log.write(json.dumps(entry) + "\n")
    return earlier_attempts + (entry["event"] == "attempt")


def _tag_submission_number(
    *, settings: Settings, experiment_name: str, fold_id: str, submission_number: int
) -> None:
    """Tag the study's MLflow fold run with the attempt's number in the submission log."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = get_or_create_experiment(experiment_name)
    run_id = get_or_create_fold_run(experiment_id, get_or_create_parent_run(experiment_id), fold_id)
    MlflowClient().set_tag(run_id, SUBMISSION_NUMBER_TAG, str(submission_number))


def score_study(*, predictions: Path, study_name: str, fold_id: str, replace: bool) -> None:
    """Log the attempt, then store and score the study, then log the outcome.

    Every attempt is written to the submission log before anything else happens, so a refusal
    leaves a record. A scored study's fold run is tagged with the attempt's number.

    Args:
        predictions: Parquet file of `PowerForecast` rows.
        study_name: The study's name; see `validate_study_name`.
        fold_id: A leaderboard fold of the CV config.
        replace: Whether to overwrite an existing `study/<study_name>` partition.

    Raises:
        Exception: Whatever `_score_study` raises; the log records the exception's type and text.
    """
    settings = Settings()
    log_path = Path(settings.local_artifacts_path) / SUBMISSION_LOG_NAME
    entry: dict[str, object] = {
        "time": datetime.now(UTC).isoformat(),
        "study_name": study_name,
        "fold_id": fold_id,
        "replace": replace,
    }
    submission_number = _append_to_log(log_path=log_path, entry={**entry, "event": "attempt"})
    try:
        _score_study(
            predictions=predictions, study_name=study_name, fold_id=fold_id, replace=replace
        )
    except Exception as error:
        _append_to_log(
            log_path=log_path,
            entry={
                **entry,
                "event": "refused",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    _append_to_log(log_path=log_path, entry={**entry, "event": "scored"})
    _tag_submission_number(
        settings=settings,
        experiment_name=f"{STUDY_EXPERIMENT_PREFIX}{study_name}",
        fold_id=fold_id,
        submission_number=submission_number,
    )


def _score_study(*, predictions: Path, study_name: str, fold_id: str, replace: bool) -> None:
    """Store a study's predictions under `study/<study_name>` and score them with `metrics`.

    Args:
        predictions: Parquet file of `PowerForecast` rows.
        study_name: The study's name; see `validate_study_name`.
        fold_id: A leaderboard fold of the CV config.
        replace: Whether to overwrite an existing `study/<study_name>` partition.

    Raises:
        ValueError: If the study name or the fold is not allowed, the file lacks a row-key column or
            holds one with the wrong dtype, the file's `fold_id` disagrees, the partition already
            exists and `replace` is False, or `replace` is True and `NGED_ALLOW_REPLACE=1` is not
            set.
        MultipleModelNamesError: If the rows carry more than one `power_fcst_model_name`.
        RowKeyMismatchError: If the row keys differ from the reference experiment's.
    """
    if replace and os.environ.get(ALLOW_REPLACE_VARIABLE) != "1":
        raise ValueError(
            f"--replace overwrites an earlier submission, so it needs {ALLOW_REPLACE_VARIABLE}=1 "
            "in the maintainer's own shell."
        )
    settings = Settings()
    # Load the CV config the way the `metrics` asset does, from the environment variable alone. A
    # `CV_CONFIG_PATH` entry in the `.env` file, which `Settings` would also read, then cannot make
    # the script and the asset validate against different folds.
    cv_config = load_cv_config(
        Path(os.environ.get("CV_CONFIG_PATH", Settings.model_fields["cv_config_path"].default))
    )
    if fold_id not in cv_config.leaderboard_fold_ids:
        raise ValueError(
            f"Fold {fold_id!r} is not a leaderboard fold; choose from "
            f"{cv_config.leaderboard_fold_ids}."
        )
    experiment_name = f"{STUDY_EXPERIMENT_PREFIX}{validate_study_name(study_name)}"
    study = _open_predictions(
        predictions=predictions, experiment_name=experiment_name, fold_id=fold_id
    )

    forecasts = pt.LazyFrame.from_existing(
        pl.scan_delta(
            settings.power_forecasts_data_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        )
    ).set_model(PowerForecast)
    if not replace and _partition_has_rows(
        forecasts=forecasts, experiment_name=experiment_name, fold_id=fold_id
    ):
        raise ValueError(
            f"{experiment_name} already holds rows for {fold_id}; pass --replace to overwrite them."
        )

    require_single_model_name(study=study, group_label=f"{experiment_name}, {fold_id}")
    require_same_row_keys(
        study=study,
        reference=PopulationFilter(
            experiment_name=cv_config.reference_experiment_name, fold_id=fold_id
        ).apply(forecasts),
        group_label=f"{experiment_name}, {fold_id}",
        reference_label=cv_config.reference_experiment_name,
        series_batch_size=SERIES_BATCH_SIZE,
    )
    _write_in_batches(
        study=study, settings=settings, experiment_name=experiment_name, fold_id=fold_id
    )
    with DagsterInstance.ephemeral() as instance:
        materialize(
            [metrics],
            run_config=RunConfig(
                ops={
                    "metrics": MetricsConfig(
                        population_filter=PopulationFilter(
                            experiment_name=experiment_name, fold_id=fold_id
                        ),
                        evaluation_scope="leaderboard",
                    )
                }
            ),
            instance=instance,
        )


def main() -> None:
    """Re-execute with a clean environment, then score the study named on the command line."""
    if os.environ.get(CLEAN_ENVIRONMENT_MARKER) != "1":
        forwarded = (
            {ALLOW_REPLACE_VARIABLE: "1"} if os.environ.get(ALLOW_REPLACE_VARIABLE) == "1" else {}
        )
        os.execve(
            sys.executable,
            [sys.executable, *sys.argv],
            {**clean_environment(os.environ), **forwarded, CLEAN_ENVIRONMENT_MARKER: "1"},
        )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("predictions", type=Path, help="Parquet file of PowerForecast rows.")
    parser.add_argument("study_name", help="The study's name, without the study/ prefix.")
    parser.add_argument("fold_id", help="A leaderboard fold id of conf/cv/default.yaml.")
    parser.add_argument("--replace", action="store_true", help="Overwrite an existing submission.")
    arguments = parser.parse_args()
    if not arguments.predictions.is_file():
        parser.error(f"{arguments.predictions} is not a regular file.")
    # The file is read several times, and the caller may change it between reads. Read a private
    # copy, in a directory only this user can write to. The directory sits beside the script, on
    # disk, because `/tmp` is memory-backed and a full fold's file is several gigabytes.
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as private_directory:
        private_copy = Path(private_directory) / "predictions.parquet"
        shutil.copyfile(arguments.predictions, private_copy)
        score_study(
            predictions=private_copy,
            study_name=arguments.study_name,
            fold_id=arguments.fold_id,
            replace=arguments.replace,
        )


if __name__ == "__main__":
    main()
