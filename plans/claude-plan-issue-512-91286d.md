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

- the whole test suite passes unchanged — **583 passed, 1 skipped** (re-measured on the tree
  with `main` merged in);
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
   `_resolve_forecaster_config` — would in fact cover every *construction* path in the repo, since
   there is no direct `Config(**kwargs)` in production code at all
   (`scripts/run_baseline_experiment.py:78-97` goes through `RegisterExperimentConfig` and the
   resolver; `cv_assets.py:400` passes an already-built config object). What it would not cover is
   **deserialisation**: `XGBoostForecaster.load`
   (`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py:240`) and
   `load_experiment_forecaster` (`packages/ml_core/src/ml_core/_mlflow_runs.py:56`) would both
   stay lenient, so a stale `meta.json` or a stale MLflow `config` tag would keep silently falling
   back to defaults. That is the decisive argument for putting the rule on the class — one
   declaration, no way to route around it, and it is the naming-poka-yoke form. This is the
   issue's own proposal; recorded here because the alternative is the obvious one a reviewer will
   ask about.

## What changes, file by file

### `packages/ml_core/src/ml_core/base_forecaster.py`

- Import `ConfigDict` from `pydantic` alongside `BaseModel, field_serializer`.
- On `BaseForecasterConfig`, add `model_config = ConfigDict(extra="forbid")` above the field
  declarations. Inherited by every subclass (`XGBoostConfig` today), so a future forecaster config
  gets the strictness without opting in.
- Add **two or three sentences** to the class docstring — not a paragraph of rationale. State what
  the setting guarantees (an unknown key raises at construction instead of being dropped, so a
  misspelled hyperparameter cannot register), state the consequence a caller must not assume away
  (the round-trip is now strict in both directions: `model_dump(mode="json")` emits exactly the
  declared fields, so re-validating a config dumped by the *same* class version always succeeds,
  while a *stored* config the current code no longer declares is refused rather than silently
  losing a field), and link to the rendered
  `ml_experimentation/model-configuration/` page for the *why*. The reasoning belongs on that page
  and nowhere else — see "Compliance with the code-style rules main just added" below.
  `Settings` in `packages/contracts/src/contracts/settings.py:405` already sets `extra="forbid"`,
  so this follows existing repo precedent.

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
- Update `RegisterExperimentConfig.config_overrides`'s `Field(description=...)` (lines 40–46).
  This string is what Dagster renders in the launchpad run-config dialog — the one place an
  experimenter actually reads what an override may contain — and it currently says only
  "Key-value overrides applied to the base YAML's model_params, replacing whole values". Add that
  an override must name a field the config class declares.

No other production code changes. The two deserialisation sites — `XGBoostForecaster.load`
(`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py:240`) and
`load_experiment_forecaster` (`packages/ml_core/src/ml_core/_mlflow_runs.py:56`) — read state that
was *persisted* by an earlier version of the class, so "same class, so the round-trip is safe" is
not an argument the test suite can make (it only ever exercises same-process round-trips). Checked
against the actual persisted state instead: `data/production_model/meta.json`'s `model_params` and
both `config` experiment tags in `mlflow.db` (experiment ids 5 and 7) contain **only** declared
`XGBoostConfig` fields — zero extras in all three. So nothing already on disk is invalidated. Note
that fields *have* been removed from `BaseForecasterConfig` in the past (`model_family`,
`power_fcst_model_name`, `power_fcst_model_version`, `task`), which is exactly how a stale key
would arise in future; see the design-philosophy check for why raising is then the right answer.

## Compliance with the code-style rules main just added

`main` was merged into this branch after the plan was first written and reviewed, bringing two new
rules in `docs/architecture/code-style.md` and a prose-style section in `CLAUDE.md`. All three bear
directly on this change, because most of its diff is docstrings and docs.

- **Spell a docs link as its rendered URL, never a repo path.** Every link this change adds to a
  docstring or a `#` comment must be written as
  `<https://openclimatefix.github.io/nged-substation-forecast/...>`. That covers the
  `BaseForecasterConfig` docstring link above and anything the `jobs.py` docstring rewrites add.
- **One home per argument.** A design decision's rationale lives on one docs page and the docstring
  links to it. The rationale for `extra="forbid"` — the unattended-search failure mode, the
  principle-8 argument — has exactly one home: the "Tweaking a config for an experiment" section of
  `docs/ml_experimentation/model-configuration.md`. The three docstrings (`BaseForecasterConfig`,
  `_UNOVERRIDABLE_MODEL_PARAMS`, `_resolve_forecaster_config`) state the rule and link there; none
  of them restates the argument. This is a change from the plan's first draft, which had the
  `BaseForecasterConfig` docstring carrying a paragraph of *why*.
- **Prose style.** The docs edits are the visible half of this change, so they answer to the new
  `CLAUDE.md` rules: name the actual thing (`n_estimtors` misspelled as a concrete example beats
  "an invalid key"), and cut whole sentences rather than clipping words.

Nothing else main brought in disturbs the plan. `checks.py` grew a late-series table cap and
`docs/live_service/operations.md` grew a section about reading `n_late` — neither touches config
validation, and `_read_promoted_model_facts` still reads `meta.json` as raw JSON inside its degrade
handler, so the "no warning path can now raise" finding still holds. Line references cited
throughout this plan were re-checked against the merged tree.

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
   `_resolve_forecaster_config(_BASE_CONFIG, {}, "exp")` succeeds *and* that every key of the real
   `conf/model/xgboost.yaml`'s `model_params` is a declared `XGBoostConfig` field. **The
   comparison must run on `model_params` after `_target_` has been popped** — take
   `_required_targets(yaml.safe_load(...), path)`'s third return value
   (`src/nged_substation_forecast/defs/jobs.py:99`) rather than the raw YAML mapping. Verified:
   the raw mapping is *not* a subset (`extra keys: {'_target_'}`), so an implementer writing the
   naive comparison gets a red test. This one **passes on `main`** and is a regression guard for
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
   `main`*: the value is `None` today, for both `BaseForecasterConfig` and `XGBoostConfig`,
   verified above. What it catches is a future forecaster config that **re-declares**
   `extra="ignore"`/`"allow"`. It is not guarding against a subclass merely setting its own
   `model_config` — pydantic v2 *merges* parent config into child, verified: a subclass declaring
   `ConfigDict(frozen=True)` still reports `extra='forbid'`. The test belongs in this module
   because it is already the place where cross-subclass config invariants are enforced, and for
   the same stated reason: enforcing one means importing every concrete forecaster, a dependency
   `ml_core` itself must not take on.

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
- **`docs/live_service/operations.md`**, item 1 of the `live_forecasts` "What the asset does" list
  (line 109), which currently says the asset "Raises if the model has no trained time
  series (re-promote first)". That list is where the asset's raise conditions are documented for
  an operator, so the new one — a saved config the current code no longer declares — belongs
  there. Risk 1 below calls this behaviour change "invisible in the diff"; this page is where it
  becomes visible.
- **`tests/test_forecaster_config_serialisation.py`**'s module docstring, which declares the
  module to be about the canonical-*serialisation* invariant and says so four times in its first
  twelve lines. Adding the `extra="forbid"` test broadens its remit to config invariants that
  need every concrete forecaster imported; rewrite the docstring to say that, keeping the existing
  explanation of why the module lives in the app tier rather than in `packages/ml_core`. It is a
  docstring, so the docstring-markdown lint applies.

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
   on the neighbouring "our own bug" condition. Two further facts make this safer than it first
   looks. Nothing currently persisted would be rejected: `data/production_model/meta.json` and
   both MLflow `config` tags hold only declared fields (checked, see above). And on the production
   box the drift cannot arise at all — `Dockerfile:29-31` and `:61` copy `src/`, `packages/` and
   `data/production_model/` into the same image from the same tree, so code and saved config ship
   together. The only way to hit the new raise is a laptop holding a stale
   `data/production_model/` beside newer code, where raising is unambiguously right. Raised here
   only because it is the one behaviour change outside R&D and it is invisible in the diff.

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

A fresh sub-agent, given the issue and the plan file but none of the reasoning behind it, attacked
the plan against the code. Its verdict was that the plan is substantially correct — premise, fix,
blast radius and design-philosophy argument all hold — with seven refinements. **All seven were
independently re-verified against the code and all seven were accepted**; none was rejected. What
each changed:

1. **Departure 3's rationale was wrong.** It claimed a resolver-boundary check would leave
   `scripts/run_baseline_experiment.py` leaky. It would not — that script goes through
   `RegisterExperimentConfig` and the resolver (`scripts/run_baseline_experiment.py:78-97`), and
   there is no direct `Config(**kwargs)` in production code anywhere. Rewritten to give the
   argument that actually decides it: the two *deserialisation* sites a boundary check cannot
   reach.
2. **Proposed test 2 would have been red on `main` as written.** `conf/model/xgboost.yaml`'s
   `model_params` includes `_target_`, so the raw mapping is not a subset of
   `XGBoostConfig.model_fields`. Confirmed empirically; the test now specifies the post-pop
   mapping.
3. **The Dagster launchpad description was missing** from the update list
   (`src/nged_substation_forecast/defs/jobs.py:40-46`) — the one string an experimenter actually
   reads when filling in `config_overrides`. Added.
4. **`tests/test_forecaster_config_serialisation.py`'s module docstring** goes stale when the
   module gains a non-serialisation invariant. Added to the docs list.
5. **"Unaffected — verified by the green suite" was an overclaim** about the two deserialisation
   sites: the suite only exercises same-process round-trips, and fields *have* been removed from
   `BaseForecasterConfig` before. Replaced with a check of the actual persisted state — zero
   undeclared keys in `data/production_model/meta.json` and in both `mlflow.db` `config` tags.
6. **`docs/live_service/operations.md`** documents `live_forecasts`' raise conditions for an
   operator and was missing from the docs list. Added.
7. **The rationale for the `extra="forbid"` invariant test was wrong about pydantic.** v2 *merges*
   parent `model_config` into the child, so a subclass cannot drop the strictness by declaring its
   own config — only by re-declaring `extra` explicitly. Confirmed empirically; reworded.

The reviewer also noted that risk 1 understates its own case, because the Dockerfile ships code
and the promoted model from the same tree. Verified and folded into risk 1.

Findings the reviewer investigated and cleared, recorded so they are not re-litigated: the
blast-radius list is exhaustive (`jobs.py:157`, `forecaster.py:240`, `_mlflow_runs.py:56` are the
only places a superset dict can reach a forecaster config; `XGBoostConfig` is the only subclass;
nothing in `scripts/`, `packages/dashboard/`, `packages/notebooks/` or `src/dashboard/` constructs
one); `conf/model/xgboost.yaml` is the only model YAML; tests 1 and 4 do fail on `main`; no
asset check validates a config (`checks.py:739-778` reads `meta.json` as raw JSON inside a
degrade handler, so no warning path can now raise); and the fail-closed argument survives
`inherent-stability.md`'s rules 1 and 2 and its "missing versus wrong" distinction.
