# Plan: version the production model as a model release (#1193)

**The problem.** The maintainer will ship improvements from the ml-experiments to AWS every week or
two, and software releases (a better dashboard, say) will go out between them without a new model.
Today the repository has one version for the software (`v0.2.1`, a git tag) and a second number,
`MODEL_VERSION`, on each forecaster class. `MODEL_VERSION` is stamped into
`power_fcst_model_version` and means "bumped with each breaking change to the implementation". It
cannot name what the maintainer ships. A retrain on new data, or a change to the feature settings
in the config, changes the forecast with no change to the class, and the three forecaster classes
each hold their own `1` or `2`. There is also no record of which model release is serving, which
ml-experiments it contains, or which run to roll back to: `promotion.json` holds only the champion
serving now and is overwritten on every promotion, so the history sits in Dagster's run history.

**The solution.** The model release becomes an integer carried by the forecaster's saved config and
stamped into `power_fcst_model_version`, and the `MODEL_VERSION` class constant is removed. A CV run
that is not a release stamps `0`, and the release CV run, named `model-release-12`, stamps `12`. A
release log (a docs page) records each release, and the `promoted_model` asset reports the release it
promotes. The software release keeps SemVer, and a software release pins
exactly one model release because the image contains it.

## Verdict, size, and departures

- **Verdict: worth implementing, as the maintainer specified.** No premise of the issue is stale:
  `grep` finds `MODEL_VERSION` in three forecaster classes and `power_fcst_model_version` in 14
  files, and `promotion.json` is overwritten per promotion.
- **Size: complex.**
    - What gets stored: fires. A new field in the saved `meta.json` config and in the MLflow `config`
      tag and params, a changed meaning for a stored `PowerForecast` column, and a release log.
    - Production serving path: fires. `XGBoostForecaster.predict` stamps `power_fcst_model_version`
      on every live forecast row.
    - Degradation rule: does not fire. The change adds no failure of a missing or stale input.
    - More than one defensible design: fires. The number could live in the config, in a file beside
      the model, or in `promotion.json`.
    - Callers not nameable without searching: fires weakly. The three `predict` stamping sites and
      their tests are known, but every `PowerForecast` fixture hard-codes a `1`.
- **Reviews: both plan reviews, the simplicity review (Sonnet) and the correctness review (Opus).**
  The diff gets all three `implement-issue` diff reviews, including the mutation pass.
- **Departures from the maintainer's summary:** none in design. The plan adds that the release log
  records the MLflow run id of the promoted fold run, because rollback needs it. The simplicity
  review removed the `promotion.json` field and the `design-evidence.md` column that the first draft
  had, and found a caller the first draft missed (`studies/quick_wins/report_production_arm.py`).

## Solution

### Walk through

#### The happy path

To ship a release, the maintainer folds the supported ml-experiments' winners into the production
pipeline code and registers a release CV run: `register_experiment_job` with `experiment_name`
`model-release-12` and `config_overrides` that include `model_release: 12`. `_forecaster_from_yaml`
builds the config through `CONFIG_CLASS(**model_params)`, so the new
`BaseForecasterConfig.model_release` field validates there, and `_identity_tags` stores it in the
`config` tag. The release number is therefore part of that CV run's identity, and re-registering
the run with a different number is rejected like any other config change.

The CV assets train and score the folds. `XGBoostForecaster.save` writes the config, including
`model_release`, into `meta.json`, and `save_to_mlflow` archives it. At predict time
`XGBoostForecaster.predict` stamps `power_fcst_model_version` from `self.model_params.model_release`
(Int16) in the place where it stamps `experiment_name` and `ml_flow_experiment_id` from the same
config, so no row can carry a version that its model's config does not.

The maintainer picks the champion fold run off the leaderboard and materialises `promoted_model`
with its run id. `promoted_model` reads back the promoted `meta.json`, as it does today for
`experiment_name`, and reports `model_release` as output metadata, so the maintainer can confirm
the right release landed. The maintainer builds the image, tags it as today, and adds a
row to the release log (`docs/live_service/model-releases.md`): the release number, the date, the
git tag `model-12`, the run's `experiment_name`, the MLflow run id of the promoted fold run (the
rollback target), the software release (`v0.2.2`) that first shipped it, and the ml-experiments it
incorporated, each linked. The same pull request adds the `design-evidence.md` rows and `informs`
fields for those ml-experiments, as the `ml-experiment` skill already requires.

#### The main branches off the happy path

- **A CV run that is not a release** (every experiment-style CV run and every leaderboard run) has
  `model_release` at its default `0`, so its rows carry `power_fcst_model_version = 0`.
- **A baseline forecaster** (`ManualHeuristicForecaster`, `ClimatologyForecaster`) stamps
  `self.model_params.model_release`, which is `0` unless a config sets it, so the baselines read
  the same field and no class constant is left.
- **A software release with no new model** keeps the image's model, so its release-log row is the
  previous model release and only the software column changes.
- **A model saved before this change** has no `model_release` in `meta.json`, and loads with the
  default `0`.

#### The main error-handling paths

- **Promotion of a release-0 model.** `promoted_model` reports `model_release: 0` and promotes. It
  refuses nothing, because promoting an unnumbered model is legitimate during development. The
  runbook says to promote only a numbered model for a deployment.
- **Re-registering an existing CV run.** A CV run registered before this change has a stored
  `config` tag without `model_release`. `_config_differences` reports a field present on only one
  side as a difference, so re-registering that run is rejected with `config.model_release: <absent>
  -> 0`. See "Risks".
- **A mistyped release number** (for example `model_release: 12` on `model-release-11`) is not
  checked. The release log is where the pairing is recorded, and a reviewer reads it.

## What changes, file by file

- **`packages/ml_core/src/ml_core/base_forecaster.py`.** Add `model_release: int = 0` to
  `BaseForecasterConfig`, with a docstring saying `0` means "not a release". Remove the
  `MODEL_VERSION: ClassVar[int]` declaration and update the class and config docstrings, which say
  that `MODEL_NAME` and `MODEL_VERSION` are class-level and stamped from the class.
- **`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`.** Remove
  `MODEL_VERSION = 2`. In `predict`, stamp `power_fcst_model_version` from
  `cfg.model_release`. Update the README paragraph at `packages/xgboost_forecaster/README.md` that
  says model-family identity is `MODEL_NAME` and `MODEL_VERSION`.
- **`packages/baseline_forecasters/src/baseline_forecasters/climatology.py` and
  `manual_heuristic.py`.** Remove `MODEL_VERSION = 1`, and stamp from the config.
- **`src/nged_substation_forecast/defs/promotion_assets.py`.** `promoted_model` adds `model_release`
  (from `model_params.get("model_release", 0)`) to its output metadata.
- **`studies/quick_wins/report_production_arm.py`.** Line 100 prints
  `XGBoostForecaster.MODEL_VERSION`, which the plan removes. The sentence drops its "the forecaster
  class is version N" clause, because the stamped versions it prints are the evidence.
- **`packages/contracts/src/contracts/power_schemas.py`.** Change the description of
  `power_fcst_model_version` to the model release: `0` for a CV run that is not a release. No dtype
  change, and no change to the other fields.
- **Docs.**
    - `docs/live_service/model-releases.md` (new): the release log and the two-identifier rule.
    - `docs/live_service/operations.md`: the promotion steps report `model_release`, and the
      rollback recipe reads the release log for the run id instead of Dagster's run history.
    - `CLAUDE.md` ("ML Model Interface"), `docs/architecture/overview.md`,
      `docs/design-philosophy/design-principles.md`, `docs/roadmap/delivery-tables.md`,
      `packages/contracts/README.md`, and the stale `MODEL_VERSION = 1` line in
      `docs/roadmap/metrics-and-leaderboard.md` (a planned persistence forecaster): replace the
      "`MODEL_VERSION` is a class constant" account.
    - `.claude/skills/ml-experiment/SKILL.md`: one sentence in "When a winner reaches production"
      saying that the release log gets a row too.
    - `mkdocs.yml`: nav entry for the new page.
- **Tests.** The `PowerForecast` fixtures that hard-code `power_fcst_model_version: 1` stay as they
  are, because the field's dtype and range are unchanged.

## Design-philosophy check

- **Production path.** `XGBoostForecaster.predict` is on the serving path. The change replaces one
  constant with one config attribute that `predict` already reads for `experiment_name` and
  `ml_flow_experiment_id`, adds no input that can be absent, and so adds no new failure and no new
  degradation path (inherent stability, H1). A model saved without the field loads with `0`.
- **`promoted_model` and the registration job fail fast,** as they do today: neither catches
  anything, and the plan adds no catch.
- **Telemetry.** Stamping the release on every row lets a forecast be traced to its release.
  Putting `model_release` on Sentry events from the live service (design principle 16) is not in
  this plan; see "Risks".
- **Hypotheses and principles.** No hypothesis changes. The plan trades one principle only:
  the model-family identity is no longer purely a code property, which the config already is for
  CV run identity.

## Tests

Each new test asserts what fails on `main` today.

- `packages/xgboost_forecaster/tests/test_forecaster.py`: a forecaster trained with
  `model_release=7` stamps `power_fcst_model_version == 7` on every row, and one with the default
  stamps `0`. On `main`, the stamp is the class constant `2`.
- `packages/baseline_forecasters/tests/test_climatology.py` and `test_manual_heuristic.py`: the same two assertions for both baselines. On `main`
  the baselines stamp `1`.
- `packages/ml_core/tests/test_base_forecaster.py`: `model_release` round-trips through
  `save`/`load` (read back from `meta.json`), and a `meta.json` without the field loads with `0`.
  Existing tests that define `MODEL_VERSION = 1` on a stub subclass are updated.
- The `PowerForecast` contract test already covers the Int16 dtype; no new test.

## Docs to update

The files listed above, each written for the present. This issue does not complete a roadmap item,
so there is no ship-time triage. The release log starts empty ("None yet"), as the evidence page
does, because no release has been numbered.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check
uv run pytest
uv run pytest --run-studies packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict    # and read the rendered release log
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files
```

## Risks and open questions

1. **Existing registered CV runs.** `_config_differences` treats a field present on one side only as
   a change, so a CV run registered before this change is rejected on re-registration until it is
   given a new name. No serious model has been trained, and a retrain costs less than a migration
   path, so I recommend accepting this. The alternative is to compare the stored config after
   filling defaults through `CONFIG_CLASS`, which is a larger change to the identity check.
2. **Release number and CV run name can disagree.** `model_release` in the config and the name
   `model-release-N` are two inputs. I recommend no check, because the release log is the pairing and
   a reviewer reads it. A check would add a rule about names to the registration job.
3. **Sentry tag.** Tagging live-service events with `model_release` is the natural next step and
   needs a look at where the live assets attach tags. I recommend a follow-up issue.
4. **Image tag.** The image tag stays as the runbook has it (derived from the run id). Adding
   `model-12` to the tag is a runbook change the maintainer can make with no code.
5. **Rows already stored.** Old rows keep their `1` and `2`, which read as releases 1 and 2. I
   recommend accepting this, because no serious model has been trained.
6. **`MODEL_VERSION` removal is a breaking change** to any code naming it. `grep` finds only the three
   classes, their tests, and docs, all updated here.

## Findings considered and rejected

- **Drop the git tag `model-N`** (simplicity review): rejected. The maintainer accepted tags when
  the identifiers were agreed, and the tag pins the commit for a release in the log.
- **Put the release log in `operations.md` instead of a new page** (simplicity review): rejected. A
  growing table in a runbook rots, and the page costs one nav line.
- **Fill stored-config defaults in `_config_differences`** (the review's optional free fix): not
  adopted. Risk 1 stays as the maintainer's accepted cost of a retrain.
