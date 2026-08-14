"""Manually launched Dagster jobs for experiment lifecycle management.

These are jobs (not assets) because they manage MLflow state and the ``cv_experiment_folds``
dynamic partition set rather than producing a data artifact. Their ops use ``OpExecutionContext``
(not ``AssetExecutionContext``) because they call ``context.instance.add_dynamic_partitions``,
which needs the Dagster instance.
"""

import json
from pathlib import Path
from typing import Any, Final, Literal, cast

import mlflow
import yaml
from contracts.config_schemas import CvConfig, class_target, import_class, load_cv_config
from contracts.settings import PROJECT_ROOT, Settings
from dagster import Config, OpExecutionContext, job, op
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from ml_core.cv_helpers import CV_PARTITION_KEY_SEPARATOR, flatten_config
from ml_core.mlflow_runs import get_or_create_experiment, get_or_create_parent_run
from ml_core.repro import provenance_tags
from mlflow.tracking import MlflowClient
from pydantic import Field

from nged_substation_forecast.defs.cv_assets import CV_EXPERIMENT_FOLDS_NAME


class RegisterExperimentConfig(Config):
    """Run config for ``register_experiment_job``."""

    experiment_name: str = Field(
        description=(
            "Human-readable, unique name; becomes the MLflow experiment name and the partition"
            " key prefix. For sweeps, generate programmatically, e.g. 'xgboost_lr0p01_depth3'."
        ),
    )
    base_model_config: str = Field(
        description="Path relative to PROJECT_ROOT, e.g. 'conf/model/xgboost.yaml'.",
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value overrides applied to the base YAML's model_params, replacing whole values,"
            " e.g. {'selected_features': ['lag_1h'], 'n_estimators': 300}. Every key must name a"
            " field the config class declares; an unknown one is rejected, so a misspelling fails"
            " here rather than training on the YAML's value."
        ),
    )
    run_mode: Literal["smoke_test", "full_cv", "register_only"] = Field(
        default="smoke_test",
        description=(
            "smoke_test: add the non-leaderboard dev folds; full_cv / register_only: add the"
            " leaderboard folds from conf/cv/default.yaml. No assets are materialised by the job"
            " itself."
        ),
    )
    description: str = Field(default="", description="Stored as an MLflow experiment tag.")


_UNOVERRIDABLE_MODEL_PARAMS: Final[dict[str, str]] = {
    "_target_": (
        "it names the forecaster class, and the config class follows from it, so base_model_config"
        " is what fixes both — point that at a different YAML instead"
    ),
    "experiment_name": "it is set from the job's own experiment_name parameter",
}
"""The ``model_params`` keys a run's ``config_overrides`` may not set, mapped to why not.

Both are ordinary keys as far as the override merge is concerned, and each needs rejecting for its
own reason. ``experiment_name`` is a *declared* field, so ``BaseForecasterConfig``'s
``extra="forbid"`` cannot see it: an override of it validates cleanly and is then overwritten by
the assignment that stamps the job's own value, discarding it without a word. ``_target_`` would be
caught by ``extra="forbid"`` unaided, and is listed here only for the message — pydantic would say
just "extra inputs are not permitted", where this names the key and points at ``base_model_config``
as the way to change the forecaster.
"""


def _forecaster_target_and_params(raw: Any, config_path: Path) -> tuple[str, dict[str, Any]]:
    """Pull the ``_target_`` path and the ``model_params`` mapping out of a parsed model YAML.

    ``base_model_config`` is free text on the launchpad, so a typo points this at the wrong file —
    ``conf/cv/default.yaml``, say, or a file that does not parse to a mapping at all. Reading both
    required pieces in one place means every such mistake reports the path and the whole expected
    shape, instead of a bare ``KeyError``.

    Args:
        raw: Whatever ``yaml.safe_load`` returned for the model YAML — ``Any``, because
            that is genuinely all a parsed YAML file is known to be until checked here.
        config_path: The file it came from, named in the error message.

    Returns:
        The forecaster's ``_target_`` and the ``model_params`` mapping, ready to splat into the
        forecaster's ``CONFIG_CLASS``.

    Raises:
        ValueError: The file is not a mapping, is missing ``_target_`` or ``model_params``, or has
            a ``_target_`` that is not a string.
    """
    try:
        forecaster_target = raw["_target_"]
        model_params = dict(raw["model_params"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{config_path} is not a usable model config: it must be a mapping with a top-level"
            " '_target_' naming the BaseForecaster subclass, and a 'model_params' mapping."
            f" Got: {error!r}"
        ) from error
    # The lookup above catches an absent `_target_`, not a present-but-not-a-string one: an empty
    # `_target_:` parses to None and a number stays a number. Either reaches import_class, which
    # fails on `str.rpartition` — naming neither the file nor the key.
    if not isinstance(forecaster_target, str):
        # TRY004 wants a TypeError, but this is one of several ways the *file's contents* can be
        # unusable, and a caller should need to catch only one exception type to mean "bad config".
        raise ValueError(  # noqa: TRY004
            f"{config_path} is not a usable model config: '_target_' must be a string naming the"
            f" BaseForecaster subclass; got {forecaster_target!r}."
        )
    return forecaster_target, model_params


def _resolve_forecaster_config(
    base_model_config: str,
    config_overrides: dict[str, Any],
    experiment_name: str,
) -> tuple[type[BaseForecaster], BaseForecasterConfig]:
    """Build the concrete forecaster class + config from a model YAML and overrides.

    Loads the base model YAML, applies ``config_overrides`` to its ``model_params``, then
    constructs the forecaster's ``CONFIG_CLASS``, which is where pydantic validates the
    hyper-parameters. Reaching the config class through the forecaster is what stops a registration
    validating hyper-parameters against a different class from the one ``load`` will rebuild them
    with.

    Args:
        base_model_config: Path relative to ``PROJECT_ROOT`` of the base model YAML.
        config_overrides: Overrides applied to ``model_params`` as **whole-value replacement**: an
            override replaces the base value outright rather than merging into it. Lists are
            replaced, not extended, and a value that is itself a mapping is replaced whole, so
            an override must restate every key of that mapping it wants to keep. Every key must
            name a field the config class declares, and the keys in
            ``_UNOVERRIDABLE_MODEL_PARAMS`` are refused on top of that — they are the resolver's
            own to set.
        experiment_name: Stamped onto the resolved config's ``experiment_name`` field.

    Returns:
        A ``(forecaster_cls, forecaster_config)`` tuple.

    Raises:
        ValueError: ``config_overrides`` names an unoverridable key, or the YAML is not a usable
            model config.
        ValidationError: An override names an undeclared key, or gives a declared one a value of
            the wrong type. Raised by the config class before any fold is scheduled. A subclass of
            ``ValueError``, so a caller that wants "the config is unusable" catches that alone.
    """
    for key, reason in _UNOVERRIDABLE_MODEL_PARAMS.items():
        if key in config_overrides:
            raise ValueError(f"config_overrides may not override {key!r}: {reason}.")
    config_path = PROJECT_ROOT / base_model_config
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    forecaster_target, model_params = _forecaster_target_and_params(
        raw=raw, config_path=config_path
    )
    model_params.update(config_overrides)
    forecaster_cls = cast(type[BaseForecaster], import_class(forecaster_target))
    forecaster_config = forecaster_cls.CONFIG_CLASS(**model_params)
    forecaster_config.experiment_name = experiment_name
    return forecaster_cls, forecaster_config


IdentityTagType = Literal["config", "forecaster_target"]
"""Type annotation for the experiment tags that together define an experiment's identity."""

IDENTITY_TAGS: Final[tuple[IdentityTagType, ...]] = ("config", "forecaster_target")
"""Runtime tuple — iterated when comparing a re-registration against the stored identity.

The ``description`` tag is deliberately absent: prose about an experiment is not part of what
makes it that experiment, so it stays freely editable by re-registering.
"""

IdentityTagsType = dict[IdentityTagType, str]
"""An experiment's identity, as the tag values that carry it."""


class ExperimentIdentityChangedError(ValueError):
    """Raised when re-registering an existing experiment under a changed config.

    A distinct subclass so a caller (or a test) can tell "this experiment name is already taken by
    a different config" apart from the ordinary ``ValueError``s raised while resolving the config
    itself.
    """


def _identity_tags(
    forecaster_cls: type[BaseForecaster], forecaster_config: BaseForecasterConfig
) -> IdentityTagsType:
    """Return the experiment tags that define this experiment's identity.

    Args:
        forecaster_cls: The resolved forecaster class.
        forecaster_config: The resolved config, with ``experiment_name`` already stamped.

    Returns:
        The ``config``/``forecaster_target`` tag values. ``config`` is the canonical JSON dump —
        see ``BaseForecasterConfig`` on why serialisation must be order-stable for this comparison
        to mean anything.
    """
    return {
        "config": forecaster_config.model_dump_json(),
        "forecaster_target": class_target(forecaster_cls),
    }


_ABSENT: Final[object] = object()
"""Sentinel for a config field present on only one side of an identity comparison.

A sentinel rather than ``None``, so a field that is genuinely ``null`` on one side and missing on
the other is reported as the difference it is.
"""


def _render_value(value: object) -> str:
    """Render one config value for the rejection message."""
    return "<absent>" if value is _ABSENT else repr(value)


def _config_differences(stored_json: str, requested_json: str) -> list[str]:
    """Return one line per differing config field, or nothing if the two configs agree.

    Compares the *parsed* JSON, not the raw strings, so a difference in key order or JSON
    formatting is not mistaken for a config change.
    """
    stored: dict[str, Any] = json.loads(stored_json)
    requested: dict[str, Any] = json.loads(requested_json)
    return [
        f"  config.{field}: {_render_value(stored.get(field, _ABSENT))}"
        f" -> {_render_value(requested.get(field, _ABSENT))}"
        for field in sorted(stored.keys() | requested.keys())
        if stored.get(field, _ABSENT) != requested.get(field, _ABSENT)
    ]


def _target_difference(tag: IdentityTagType, stored: str, requested: str) -> list[str]:
    """Return the difference line for a class-target tag, or nothing if it is unchanged."""
    return [] if stored == requested else [f"  {tag}: {stored!r} -> {requested!r}"]


def _reject_changed_identity(
    experiment_name: str, stored_tags: dict[str, str], requested_tags: IdentityTagsType
) -> None:
    """Raise if re-registering ``experiment_name`` would change its identity.

    An experiment's identity **is** its config: every fold of an experiment must be trained and
    scored under one config, or its leaderboard row silently mixes two different models. Folds
    already materialised under the old config cannot be un-trained, and the assets read the config
    back from the experiment tag (``load_experiment_forecaster``), so re-pointing that tag
    mid-flight would poison every comparison built on the experiment. A changed config is therefore
    a *new* experiment, and the only correct answer is to reject it — before anything has been
    written.

    Called with the experiment's stored tags before any write, so a rejection leaves MLflow exactly
    as it found it. A tag absent from ``stored_tags`` is not a change: that is the untagged
    experiment ``get_or_create_experiment`` creates as its self-healing fallback, which this
    registration is entitled to complete.

    The decision to reject *is* the list of differences to report, so the error can never claim a
    change it cannot then name.

    Args:
        experiment_name: The MLflow experiment name, for the error message.
        stored_tags: The existing experiment's tags (empty for a brand-new experiment).
        requested_tags: The identity this registration is asking for, from ``_identity_tags``.

    Raises:
        ExperimentIdentityChangedError: If any identity tag is already set to a different value.
    """
    differences = [
        line
        for tag in IDENTITY_TAGS
        if tag in stored_tags
        for line in (
            _config_differences(stored_json=stored_tags[tag], requested_json=requested_tags[tag])
            if tag == "config"
            else _target_difference(tag=tag, stored=stored_tags[tag], requested=requested_tags[tag])
        )
    ]
    if not differences:
        return
    raise ExperimentIdentityChangedError(
        f"MLflow experiment {experiment_name!r} is already registered with a different"
        " configuration. An experiment's identity is its configuration — every fold must be"
        " trained and scored under the same one — so a changed configuration is a new experiment."
        " Register it under a new experiment_name.\nDifferences (registered -> requested):\n"
        + "\n".join(differences)
    )


def _fold_ids_for_run_mode(run_mode: str, cv_config: CvConfig) -> list[str]:
    """Return the fold ids a run mode expands to (read from the CV config, never hard-coded).

    Selection keys off the ``leaderboard`` flag, not fold position: ``smoke_test`` uses the
    non-leaderboard dev folds; ``full_cv`` and ``register_only`` use the leaderboard folds.
    """
    if run_mode == "smoke_test":
        return [fold.fold_id for fold in cv_config.folds if not fold.leaderboard]
    return cv_config.leaderboard_fold_ids


@op
def register_experiment(context: OpExecutionContext, config: RegisterExperimentConfig) -> None:
    """Create the MLflow experiment + parent run and add the experiment's CV partition keys.

    Idempotent for the *same* config: re-running with the same ``experiment_name`` resolves the
    existing experiment, parent run, and partition keys rather than duplicating them, and may
    freely update the ``description`` and add the other run mode's folds. Re-running with a
    **changed** config is rejected outright — see ``_reject_changed_identity``. Materialises no
    assets — the user materialises ``trained_cv_model`` / ``cv_power_forecasts`` for the new
    partitions afterwards.

    Raises:
        ExperimentIdentityChangedError: If ``experiment_name`` is already registered under a
            different config. Raised before any MLflow write, so the experiment is left exactly
            as it was.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    forecaster_cls, forecaster_config = _resolve_forecaster_config(
        base_model_config=config.base_model_config,
        config_overrides=config.config_overrides,
        experiment_name=config.experiment_name,
    )
    identity_tags = _identity_tags(
        forecaster_cls=forecaster_cls, forecaster_config=forecaster_config
    )

    # get_or_create_experiment writes nothing when the name already exists, so the check below
    # still runs before this registration's first write. On a brand-new name it creates the
    # (untagged) experiment, which has no stored identity to conflict with.
    experiment_id = get_or_create_experiment(config.experiment_name)
    client = MlflowClient()
    _reject_changed_identity(
        experiment_name=config.experiment_name,
        stored_tags=mlflow.get_experiment(experiment_id).tags,
        requested_tags=identity_tags,
    )

    # Log the params before the experiment tags. MLflow params are write-once, so this is the one
    # write the tracking store can reject; doing it first means a rejection can never leave the
    # experiment tagged with a config that its params contradict. The check above should make such
    # a rejection unreachable, but an experiment registered before that check existed may already
    # carry params that disagree with its tag, and the ordering keeps even that case one-sided.
    # log_params is not itself atomic — a batch containing one conflicting key still writes the
    # batch's other keys before raising — so this bounds the damage to the params rather than
    # eliminating it. Reaching that requires an experiment whose tags failed to write on an earlier
    # registration, which is why the tags are the last thing written.
    parent_run_id = get_or_create_parent_run(experiment_id)
    with mlflow.start_run(run_id=parent_run_id):
        mlflow.log_params(flatten_config(forecaster_config))
        mlflow.set_tags(
            {
                "model_family": forecaster_cls.MODEL_NAME,
                "weather_source": forecaster_config.weather_source,
                # Provenance: which code registered this experiment. Registration reads no data
                # tables, so no Delta versions here — those are stamped at train/predict/metrics.
                **provenance_tags("register"),
            }
        )

    # The identity tags carry the resolved config as JSON plus the forecaster class the JSON itself
    # cannot name, so assets can reconstruct the exact forecaster — and, through its CONFIG_CLASS,
    # the exact config subclass — from MLflow alone. See load_experiment_forecaster.
    for tag_name, tag_value in identity_tags.items():
        client.set_experiment_tag(experiment_id=experiment_id, key=tag_name, value=tag_value)
    client.set_experiment_tag(
        experiment_id=experiment_id, key="description", value=config.description
    )

    cv_config = load_cv_config(settings.cv_config_path)
    fold_ids = _fold_ids_for_run_mode(run_mode=config.run_mode, cv_config=cv_config)
    partition_keys = [
        f"{config.experiment_name}{CV_PARTITION_KEY_SEPARATOR}{fold_id}" for fold_id in fold_ids
    ]
    context.instance.add_dynamic_partitions(
        partitions_def_name=CV_EXPERIMENT_FOLDS_NAME, partition_keys=partition_keys
    )

    context.add_output_metadata(
        {
            "experiment_name": config.experiment_name,
            "mlflow_experiment_id": experiment_id,
            "run_mode": config.run_mode,
            "n_partition_keys": len(partition_keys),
            "partition_keys": str(partition_keys),
        }
    )


@job
def register_experiment_job() -> None:
    """Register a new experiment: create its MLflow records and CV partition keys."""
    register_experiment()
