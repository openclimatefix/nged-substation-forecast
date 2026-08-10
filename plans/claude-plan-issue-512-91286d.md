# Plan — issue #512: reject unknown keys in `config_overrides`

> 🤖 **This plan was written by [Claude Code](https://claude.com/claude-code)**, acting on Jack's
> behalf.

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/512>
Branch: `claude/plan-issue-512-91286d`

## Verdict

**Worth implementing, as described.** The premise was verified against `main`, not taken on trust:

```text
XGBoostConfig(selected_features={'a'}, n_estimtors=5000).n_estimators  ->  1000
hasattr(config, 'n_estimtors')                                         ->  False
BaseForecasterConfig.model_config.get('extra')                         ->  None  (pydantic default 'ignore')
```

The typo is dropped without a word and the run trains on the base YAML's value. Both planned
search mechanisms (the LLM auto-research agent, the training-history variant grid) are unattended,
so nobody reads the resolved config before a grid of identical runs lands on the leaderboard with
plausible scores — a [principle 8](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#8-every-experiment-is-scored-identically)
failure, not an ergonomics one.

The fix was also verified rather than assumed. With `extra="forbid"` applied as a throwaway patch:

- the whole test suite passes unchanged — **579 passed, 1 skipped**;
- `_resolve_forecaster_config("conf/model/xgboost.yaml", {"n_estimtors": 5000}, "exp")` raises
  `ValidationError: n_estimtors — Extra inputs are not permitted`.

The patch was reverted; this branch carries no code.

## Departures from the issue body

1. **The issue's "check before landing" is now discharged, and half of it is stale.** The YAML
   check holds: every `conf/model/xgboost.yaml` `model_params` key (`selected_features`,
   `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`,
   `colsample_bytree`, `device`, `objective`, `weather_source`, `training_strategy`) is a declared
   field on `XGBoostConfig` or `BaseForecasterConfig`, and `_target_` is popped by
   `_required_targets` before construction. The "two `_target_` keys" phrasing is stale, as the
   issue comment notes: since [#514](https://github.com/openclimatefix/nged-substation-forecast/pull/514)
   an *override* of `_target_` is refused by `_UNOVERRIDABLE_MODEL_PARAMS` and never reaches the
   constructor at all. Only the YAML's own `_target_` is popped.

2. **Both halves of `_UNOVERRIDABLE_MODEL_PARAMS` stay**, and the plan says why for each — the
   issue comment asked for this to be a conscious call rather than an accident.

    - `experiment_name` **must** stay. It is a declared field on `BaseForecasterConfig`, so
      `extra="forbid"` cannot see it; an override validates cleanly and is then overwritten by
      `forecaster_config.experiment_name = experiment_name`. Verified: `XGBoostConfig(
      selected_features={'a'}, experiment_name='foo')` constructs fine under `extra="forbid"`.
      This is the same silent-discard failure the issue is about, arriving by a route the issue's
      change cannot close.
    - `_target_` stays **on the strength of its message alone**. Verified that `extra="forbid"`
      would catch it unaided (`XGBoostConfig(selected_features={'a'}, _target_='x.Y')` raises
      `extra_forbidden` on `_target_`), but pydantic's message says only "extra inputs are not
      permitted". The guard's message says which key, why it cannot be set, and that
      `base_model_config` is the way to change the config class. Keeping the two keys together
      also keeps the rule readable as one list rather than as two mechanisms that happen to
      overlap.

3. **The change is deliberately made on the class, not at the resolver boundary.** The narrower
   alternative — validate override keys against `config_cls.model_fields` inside
   `_resolve_forecaster_config` — would guard one function and leave every direct construction
   (`scripts/run_baseline_experiment.py`, tests, any future programmatic caller) as leaky as
   today. `extra="forbid"` on the base class is the naming-poka-yoke form: one declaration, no way
   to route around it. This is the issue's own proposal; recorded here because the alternative is
   the obvious one a reviewer will ask about.

## What changes, file by file

### `packages/ml_core/src/ml_core/base_forecaster.py`

- Import `ConfigDict` from `pydantic` alongside `BaseModel, field_serializer`.
- On `BaseForecasterConfig`, add `model_config = ConfigDict(extra="forbid")` above the field
  declarations. Inherited by every subclass (`XGBoostConfig` today), so a future forecaster config
  gets the strictness without opting in.
- Extend the class docstring with a short paragraph on *why* — an unknown key is almost always a
  misspelled hyperparameter, and an unattended search would otherwise register a grid of identical
  runs that all score plausibly. Note that this also makes the round-trip strict in both
  directions: `model_dump(mode="json")` emits exactly the declared fields, so
  `model_validate` / `model_validate_json` of a config dumped by the *same* class version always
  succeeds, and a stored config that no longer matches the code is refused rather than silently
  losing a field. `Settings` in `packages/contracts/src/contracts/settings.py:404` already sets
  `extra="forbid"`, so this follows existing repo precedent.

### `src/nged_substation_forecast/defs/jobs.py`

- Rewrite the `_UNOVERRIDABLE_MODEL_PARAMS` module docstring (currently lines 65–71). It says both
  keys "would be discarded without a word downstream of it: `_target_` by pydantic's
  `extra="ignore"`…" — that clause becomes false the moment this lands. Rewrite it to describe the
  present: `experiment_name` is a declared field that the resolver overwrites, so nothing else can
  catch it; `_target_` would be caught by `extra="forbid"`, and is listed here so the error names
  the key and points at `base_model_config`.
- Update `_resolve_forecaster_config`'s docstring: its `config_overrides` arg description says
  "Every `model_params` key is overridable except those in `_UNOVERRIDABLE_MODEL_PARAMS`", which
  now reads as though any *other* key is accepted. State that an override must name a declared
  field of the config class, and add `ValidationError` to the `Raises:` section.

No other production code changes. `XGBoostForecaster.load`
(`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py:238`) and
`load_experiment_forecaster` (`packages/ml_core/src/ml_core/_mlflow_runs.py:56`) both deserialise
output of `model_dump(mode="json")` / `model_dump_json()` from the same class, so they are
unaffected — verified by the green suite, which exercises both round-trips.

## Design-philosophy check

**Which tier does this run in?** Registration (`register_experiment_job` → the resolver) is R&D,
where `docs/design-philosophy/inherent-stability.md` says *fail fast*: a quietly-degraded run
poisons every comparison built on it. Raising a `ValidationError` before a single fold is
scheduled is exactly the prescribed behaviour. No asset check is added or edited, so the
`WARN`/`blocking=False`/cannot-raise rules are not engaged.

**The one production path this touches, and why raising there is correct.** `live_forecasts`
(`src/nged_substation_forecast/defs/production_assets.py:237`) loads the promoted model via
`load_forecaster_from_dir` → `XGBoostConfig.model_validate(meta["model_params"])`. The only way
that dict can carry an unknown key is a code change that removed or renamed a config field after
the model was saved. Today that key is silently dropped and the field falls back to its default,
so production would serve forecasts under a config that is not the one the model was trained
under — silently wrong output. After this change it raises. That is the right side of the ladder:
inherent-stability reserves raising for "states that are our own bug (an empty promoted model, a
contract violation)", and this asset already raises unconditionally two lines later when the
promoted model has no trained series. A promoted model whose saved config no longer matches the
code is the same class of bug. It is also cheap to recover from — CLAUDE.md's "this is a young
project" rule says a change invalidating a saved config costs a retrain, not a migration path.

**Principles.** Serves principle 8 (every experiment is scored identically) by making a bad grid
fail to parse rather than fail to be noticed, and enacts the *naming poka-yoke* habit listed under
"Industry best practices we have not yet absorbed". Nothing is traded away. The hypothesis it
protects is **H2** (a hundred experiments per person in a peak month, tested by **T2.1**): high
throughput is only worth having if the experiments are what they claim to be.

## Tests

All in `tests/test_jobs.py`, beside the existing resolver tests.

1. **`test_resolve_rejects_an_unknown_override_key`** — asserts
   `_resolve_forecaster_config(_BASE_CONFIG, {"n_estimtors": 5000}, "exp")` raises
   `ValidationError` matching `n_estimtors`. *Fails on `main` today*: verified above that the call
   returns successfully with `n_estimators == 500`, so `pytest.raises` gets no exception.

2. **`test_resolve_accepts_every_key_the_base_yaml_declares`** — asserts
   `_resolve_forecaster_config(_BASE_CONFIG, {}, "exp")` succeeds *and* that
   `set(model_params) <= set(XGBoostConfig.model_fields)` for the real
   `conf/model/xgboost.yaml`, so adding an undeclared key to the YAML fails the test rather than
   only failing at registration time. This one **passes on `main`** and is a regression guard for
   the YAML, not a test of the change — it is included because the issue's "check before landing"
   is otherwise a one-off manual check that nothing re-checks. Flagged explicitly so it is not
   mistaken for evidence.

3. Amend the docstring of the existing **`test_resolve_rejects_an_override_of_a_key_it_would_discard`**
   (`tests/test_jobs.py:99`), which asserts the same `extra="ignore"` claim as the `jobs.py`
   docstring and goes stale with it. The assertion itself stands unchanged — both keys are still
   refused by the explicit guard, and for `_target_` the docstring must now say the guard exists
   for its message, not because pydantic would let it through.

Add to `tests/test_forecaster_config_serialisation.py`:

1. **`test_every_config_class_forbids_extra_keys`** — parametrised over the existing
   `_CONFIG_CLASSES` list, asserts `config_cls.model_config.get("extra") == "forbid"`. *Fails on
   `main`*: the value is `None` today, verified above. This is the test that catches a future
   forecaster config that sets its own `model_config` and drops the inherited strictness — the
   same shape as that module's existing set-serialiser invariant, and it belongs there because
   that module is already the place where cross-subclass config invariants are enforced.

## Docs to update

- **`docs/ml_experimentation/model-configuration.md`**, the "Tweaking a config for an experiment"
  section (around line 175). It currently says "any `model_params` key can be overridden — except
  two, which are rejected rather than silently discarded". Rewrite so the rule is stated positively
  and completely: an override must name a declared field of the config class (the field tables
  directly above this section are that list), an unknown key raises `ValidationError` at
  registration before any fold runs, and `_target_` / `experiment_name` are declared-or-popped keys
  the resolver owns and refuses separately.
- **`docs/ml_experimentation/dagster-workflow.md`**, step 5's table row for `config_overrides`
  ("Merged onto `model_params` in the YAML") and the numbered "What the job does" item 2 ("pydantic
  validates every hyperparameter before anything is registered"). Item 2 becomes stronger and
  should say so: pydantic now validates the *set* of keys as well as their values.
- **`docs/architecture/overview.md`** line 22 ("so pydantic validates every hyperparameter at
  registration time") — same one-clause strengthening.
- **`conf/model/xgboost.yaml`**'s header comment, which explains the resolution mechanism and ends
  "A registered experiment's `config_overrides` replace whole values under `model_params`". Add
  that an override naming a key the config class does not declare is rejected — this file is where
  someone reads the key names, so it is where the rule is cheapest to learn.

No roadmap item is completed by this issue, so no ship-time roadmap triage (no status banner to
move, no "Implementation details" section to delete).

## Verification commands

The green-before-push set:

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check && uv run pytest
```

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

Plus, specific to this change:

```bash
uv run mkdocs build --strict
```

(four docs pages change, two of them carrying links — read the rendered HTML for the
`model-configuration.md` section, since `mkdocs-authoring` warns that list and link breakage passes
both linters).

No network-marked tests are needed: nothing here touches NWP conversion or S3.

## Risks and open questions

1. **Does the promoted-model load path becoming fail-closed need Jack's sign-off?** Recommendation:
   **no, land it as-is.** Reasoning in the design-philosophy check above — the alternative is
   serving forecasts under a config the model was not trained under, and the asset already raises
   on the neighbouring "our own bug" condition. Raised here because it is the one behaviour change
   outside R&D, and it is invisible in the diff.

2. **`CvConfig` and `CvFoldConfig` have the identical weakness — out of scope, flagged not fixed.**
   `packages/contracts/src/contracts/config_schemas.py:86,107` are plain `BaseModel`s loaded from
   `conf/cv/default.yaml` via `CvConfig.model_validate`, so a misspelled `min_training_months` in
   the CV YAML is silently dropped and every experiment on the leaderboard quietly uses the
   default. That is arguably a worse principle-8 failure than the one this issue fixes, because it
   is shared by *all* experiments rather than one grid. Per CLAUDE.md I have not touched it.
   Recommendation: **file a follow-up issue**; no open issue covers it (searched). Jack's call
   whether to fold it into this PR instead — it is a two-line change, but it widens a
   single-purpose diff.

3. **Should `RegisterExperimentConfig` (`src/nged_substation_forecast/defs/jobs.py:28`) forbid
   extras too?** Recommendation: **no.** It is a Dagster `Config`, populated from the run-config
   dialog, and Dagster validates the run config against the schema before the job starts. Noted so
   the reviewer does not read its absence as an oversight.

## Review findings

Filled in at step 5, after the adversarial review.
