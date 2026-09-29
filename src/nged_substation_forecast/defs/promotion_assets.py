"""Research Dagster assets for selecting and promoting a champion model from MLflow."""

import json
from pathlib import Path

import mlflow
from contracts.settings import Settings
from dagster import (
    AssetExecutionContext,
    Config,
    MetadataValue,
    TableRecord,
    asset,
)
from ml_core.mlflow_runs import list_promotable_runs
from ml_core.production_helpers import fetch_model_artifacts

from nged_substation_forecast.defs._tags import RESEARCH_LAYER_TAGS


@asset(tags=RESEARCH_LAYER_TAGS)
def promotable_model_runs(context: AssetExecutionContext) -> None:
    """List MLflow fold runs eligible for promotion via ``promoted_model``.

    Purely informational: materialise this on demand (it has no dependents and writes nothing to
    disk) to refresh the candidate list as a metadata table in the Dagster UI, then copy the
    champion's ``run_id`` into ``promoted_model``'s launchpad. The champion is still picked by
    eye off the MLflow leaderboard (metrics vary per experiment, so there is no single sort key
    to automate the pick) — this just saves retyping/misremembering the run id.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = list_promotable_runs()

    table = [
        TableRecord(
            {
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "fold_id": run.fold_id,
                "last_finished_at": (
                    run.last_finished_at.strftime("%Y-%m-%d %H:%M UTC")
                    if run.last_finished_at is not None
                    else "Not recorded"
                ),
            }
        )
        for run in runs
    ]
    context.add_output_metadata(
        {"n_candidates": len(runs), "candidates": MetadataValue.table(table)}
    )


class PromotedModelConfig(Config):
    """Run config for the manually-triggered ``promoted_model`` asset."""

    mlflow_run_id: str
    """The champion fold run id, picked from the MLflow leaderboard (or from
    ``promotable_model_runs``'s candidate table)."""


@asset(tags=RESEARCH_LAYER_TAGS)
def promoted_model(context: AssetExecutionContext, config: PromotedModelConfig) -> None:
    """Promote a champion model from MLflow to local disk for zero-MLflow-at-runtime inference.

    Manually triggered from the Dagster UI launchpad with ``mlflow_run_id`` set to the champion
    fold's run id — materialise ``promotable_model_runs`` first and copy the id from its output
    metadata table if you don't have it to hand. Downloads that run's saved model artifacts to
    ``Settings.production_model_path`` (via ``ml_core.production_helpers.fetch_model_artifacts``,
    which replaces the directory atomically), then reads back ``meta.json`` to report provenance.
    ``live_forecasts`` reads this directory with a plain disk load — never MLflow.

    Promotion refuses a model whose saved config this code cannot rebuild — a feature name it
    cannot parse, or a ``model_params`` key it no longer declares — and refuses it before the
    directory is replaced, so the previous champion stays in place and keeps serving.

    Every such refusal reaches the operator as a failed materialisation: this asset catches
    nothing, unlike the rest of ``defs/``. Degrading is what the production *serving* path does,
    because a late or partial forecast beats none; promotion has no such fallback, since the
    outgoing champion keeps serving whatever happens here. A promotion that half-succeeded
    quietly would be strictly worse than one that stopped and said so. The rules this follows:
    <https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules>.

    Promotion as a Dagster materialisation gives an audit trail and lineage for free, rather than
    a bare script (a script wrapper for the eventual Docker build (#222) stays trivial by calling
    the same ``fetch_model_artifacts`` helper).
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    production_model_path = Path(settings.production_model_path)
    fetch_model_artifacts(config.mlflow_run_id, production_model_path)

    meta = json.loads((production_model_path / "meta.json").read_text())
    model_params = meta.get("model_params", {})
    context.add_output_metadata(
        {
            "mlflow_run_id": config.mlflow_run_id,
            "model_class": meta.get("model_class"),
            "experiment_name": model_params.get("experiment_name"),
            "n_trained_time_series": len(meta.get("trained_time_series_ids", [])),
            "path": str(production_model_path),
        }
    )
