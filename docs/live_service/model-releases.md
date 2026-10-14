# Model releases

**This page is the log of the production model releases.** A model release is a numbered, trained
XGBoost model that the maintainer ships to the live service. The number is an integer, written
`model-12` in prose and in a git tag. It is stamped onto every forecast row as
`power_fcst_model_version`, so any stored forecast can be traced to the release that made it.

## Two version numbers

**The software and the model carry separate version numbers.** The software release is a SemVer git
tag (`v0.2.2`), and it changes when the code changes: a better dashboard, say. The model release
changes when the trained model changes, usually because an ml-experiment's winner was ported into
the production pipeline, or because the model was retrained. A software release contains exactly one
model release, because the Docker image bakes the model in. A software release with no new model
keeps the previous model release, and only the software column of the log changes.

## How to cut a model release

1. Port the supported ml-experiments' winners into the production pipeline, adding the
   [design-evidence](../architecture/design-evidence.md) rows as the `ml-experiment` skill says.
2. Register a cross-validation run named `model-release-N` whose `config_overrides` include
   `model_release: N`. The number is part of the run's registered configuration, so it travels
   with the saved model in `meta.json`.
3. Promote the champion fold run and deploy it, following [Operating the live
   service](operations.md#step-2-materialise-promoted_model). The `promoted_model` asset reports
   `model_release` as output metadata, so check the number there.
4. Tag the commit `model-N`, and add a row to the log below in the same pull request.

A cross-validation run that is not a release leaves `model_release` at its default `0`, and promoting
such a model is allowed during development. Promote only a numbered model for a deployment.
Nothing checks that the number in the config matches the run's name, so the log is the record of
the pairing.

## The log

Rollback needs the MLflow run id and the image tag, so each row records both.

| Release | Date | Git tag | MLflow run id (promoted fold run) | Image tag | Software release | Experiments incorporated |
|---|---|---|---|---|---|---|
| None yet | | | | | | |
