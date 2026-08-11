# Plan — #486: reject a promoted model whose features this code cannot parse

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/486>
Branch: `claude/reject-unservable-promoted-model-486`

## Verdict

**Worth implementing, and cheaply — but it is a preventive guard against the *next* feature
rename, not a fix for anything currently broken. The issue oversells what it buys, and the plan
says so rather than inheriting the overclaim.**

Two corrections to the issue's framing, both verified rather than argued:

### Departure 1 — this guard would not have caught the incident the issue opens with

The issue presents this as the fix for the #463 breakage. It is not.

- PR #463 (the `local_utc_offset` → `local_utc_offset_minutes` rename) merged **2026-08-07
  13:22 UTC**.
- `data/production_model/promotion.json` shows the model currently on disk was promoted
  **2026-08-07 17:35 UTC** — after the rename — and its `meta.json` carries the new name.
- The issue reports live ticks failing *before* that, which is only possible if the model being
  served had been promoted earlier, under the old vocabulary. (That is the issue's testimony, not
  something recoverable from disk: the directory holds only the current model, and a previous
  promotion leaves no trace in it.)

So #463 was **promote-then-change-the-code**. A promotion-time guard runs before the breaking
change and sees nothing wrong. What it catches is the **mirror image** — promoting a model whose
vocabulary the *current* code cannot parse. That is a genuine hazard, but a different one, and the
PR body must not claim otherwise.

### Departure 2 — the hazard is forward-looking; nothing in the store today would trip it

It is tempting to point at MLflow and say the trap is already set. `list_promotable_runs()` returns
three candidates, and two of them do carry the stale vocabulary:

| run id | experiment | vocabulary | artifacts |
|---|---|---|---|
| `7acafbe3…` | `xgboost_cv_0002` | `local_utc_offset_minutes` (current) | `model.tar.gz` |
| `aecc8c3f…` | `xgboost_no_power_lags` | `local_utc_offset` (stale) | `model/` |
| `cf8ddd29…` | `xgboost_cv_0001` | `local_utc_offset` (stale) | `model/` |

But both stale runs predate the single-archive change, so they hold the old `model/` **directory**
layout, and `_download_and_unpack_model` already refuses them with an actionable
`MlflowException` naming `model.tar.gz` — a gate that exists today and runs strictly before
anything this plan adds. Promoting either fails right now, loudly, for a different reason.

So the honest statement of value: **every run trained from now on is stored as a promotable
archive regardless of its feature vocabulary, and the next rename recreates the trap with no gate
in front of it.** Renames happen — #431/#463 is one, three months into the project. The guard costs
a few lines and one `meta.json` read; a re-promotion of a pre-rename run costs a broken live
service until someone reads a stack trace. Worth building on those terms, not on "two thirds of the
candidate list is poison".

### Departure 3 — where the check goes, and what it reads

The issue proposes validating in the `promoted_model` asset after `fetch_model_artifacts`, and
notes as an open question that validating before the atomic swap would be better. It would. The
placement here goes one step further, and deliberately does **not** load the model to do it:

**One pure function, `_check_selected_features_are_parseable(selected_features, source)`, in
`ml_core._production_helpers`. Two call sites:**

| Call site | Reads | What it buys |
|---|---|---|
| `fetch_model_artifacts`, after unpack and **before** the swap | the staged `meta.json`'s `model_params.selected_features` | Promotion refuses; `dest` is never touched, so the previous champion keeps serving |
| `load_forecaster_from_dir` | the reconstructed `forecaster.model_params.selected_features` | Every `live_forecasts` tick fails *at load* rather than deep in `engineer()`, with a message naming the feature and the remedy |

**Why `fetch_model_artifacts` reads `meta.json` directly rather than calling
`load_forecaster_from_dir` on the staging directory** (the obvious-looking alternative, which is
wrong three ways):

1. It would break two existing tests. `packages/ml_core/tests/test_base_forecaster.py`'s
   `_FakeForecaster.save` writes `{"model_class": "fake"}` (line 55–57), and `import_class("fake")`
   raises `ValueError: 'fake' is not a fully-qualified class path`. Both
   `test_fetch_model_artifacts_unpacks_the_archive_into_dest` (line 220) and the #470
   shrinking-population test (line 194) call `fetch_model_artifacts` on that fake.
2. That file's docstring says the fake exists to keep these tests "free of any model-library
   dependency", and `packages/ml_core/pyproject.toml` does not depend on `xgboost_forecaster`.
   Routing through `load_forecaster_from_dir` would force a real model into `ml_core`'s test suite.
3. It would load every booster at promotion time — several GB at V2's ~2,500 series — to answer a
   question that is answerable from one JSON field.

Reading the field directly is cheaper, keeps `ml_core`'s tests model-library-free, and matches what
the issue itself proposed (`ParsedFeatures.from_strings(set(model_params["selected_features"]))`).

The cost of not loading: promotion no longer proves `model_class` is importable or the boosters
readable. That is out of this issue's scope and is already caught at the first `live_forecasts`
tick, which is loud (the run fails and `live_forecasts_job`'s Sentry failure hook reports it).

### Departure 4 — booster `feature_names` comparison: **rejected**, with evidence

The issue's second open question asks whether to compare `Booster.feature_names` against
`selected_features`. Verified empirically against the installed xgboost 3.4.0:

- `xgb.QuantileDMatrix(polars_df, …)` sets `feature_names` from the frame's columns, and those
  names survive a `.ubj` `save_model`/`load_model` round-trip.
- `Booster.predict` validates them and raises `ValueError: feature_names mismatch: […] […] /
  training data did not have the following fields: …`.

`XGBoostForecaster._feature_cols` is `sorted(self.model_params.selected_features)`
(`forecaster.py:90-91`) and `_prepare_features` selects exactly those columns, so a
`meta.json`-versus-booster disagreement already fails loudly, naming the offending column, at the
first `predict`. It is not the *silent* class of failure this issue exists to close, and it can only
arise from hand-editing `meta.json`, which is already ruled out. Adding it needs either a new hook
on the `BaseForecaster` ABC or model-specific code in a model-agnostic helper. Not in this change.

## What changes, file by file

### `packages/ml_core/src/ml_core/_production_helpers.py`

- **New private function** `_check_selected_features_are_parseable(selected_features: set[str] |
  None, source: str) -> None`. Calls `ParsedFeatures.from_strings(...)`; on `ValueError`, re-raises
  a `ValueError` `from` it whose message names `source` (the directory or run being validated), the
  underlying parse error (which already names the offending feature), and the remedy — *re-train
  and re-promote; never hand-edit `meta.json`*. A `None` `selected_features` returns without
  checking: an absent key is not a malformed vocabulary, and `BaseForecaster.save`'s documented
  contract mandates only `model_class`.

  **Message constraint (easy to trip):** the message must not contain the substring `mlflow` in any
  casing. `scripts/build_and_verify_image.sh` hard-fails the image on `grep -qi "mlflow"` over the
  container log as its one automated hermeticity gate, and this message can reach that log. Say
  "re-promote", never "re-promote from MLflow".

  Import `ParsedFeatures` from `ml_core.features._parsed_features` — `ml_core/features/__init__.py`
  exports only `FeatureEngineer` and `TabularFeatureEngineer`, and widening that public surface for
  one internal caller is not worth it. No import cycle: `ml_core.base_forecaster` already imports
  from `ml_core.features`.

- **`fetch_model_artifacts`** — after `_download_and_unpack_model`, before `promotion.json` is
  written and before the `rmtree`/`move`: read the staged `meta.json`, pull
  `model_params.selected_features` (absent → `None`), and call the check with the run id as
  `source`. Ordering is unpack → validate → stamp → swap, so a rejected promotion writes nothing
  anywhere and `dest` is untouched. Docstring gains: the swap now gates on the model's feature
  vocabulary, and a rejected promotion leaves the previous champion in place.
- **`load_forecaster_from_dir`** — call the check on the loaded forecaster's
  `model_params.selected_features` before returning; add the `ValueError` to `Raises:`. Docstring
  gains one sentence: the returned forecaster is one this code can engineer features for, not
  merely one it could deserialise.
- **Module docstring** — it calls both disk/MLflow helpers "thin, single-purpose IO wrappers".
  Reword: they are now the gate between a saved model and this code's ability to serve it.

### `src/nged_substation_forecast/defs/production_assets.py`

No code change in either asset. Docstring edits only:

- `promoted_model` — promotion refuses a model whose `selected_features` this code cannot parse,
  and refuses it before the directory swap, so the previous champion survives.
- `live_forecasts` — the "Loads the production model…" paragraph gains the same fact, beside the
  existing empty-population raise it sits next to.

### `packages/ml_core/tests/test_base_forecaster.py`

`_FakeForecaster.save` gains `"model_params": self.model_params.model_dump(mode="json")` in the
`meta.json` it writes. That makes the fake faithful to what every real `BaseForecaster.save` writes,
and is what lets the new promotion test below be written here with no model-library dependency. The
two existing `fetch_model_artifacts` tests keep passing: the fake's default config is
`BaseForecasterConfig(selected_features=set())`, which parses fine.

## Design-philosophy check

- **Rule 1** reserves raising for "states that are our own bug — an empty promoted model, a
  contract violation". A promoted model whose feature vocabulary the serving code cannot parse is
  the same class as the empty-promoted-model raise sitting fifteen lines away in `live_forecasts`.
  It is not the outside world misbehaving; it is an artefact we produced meeting code we wrote.
  This issue's own comment makes that argument by reference to #446, and #446 makes it in the
  opposite direction for NWP staleness. Consistent.
- The failure-modes table already carries the row **"The promoted model is empty or unloadable →
  Hard failure — the asset raises → Unchanged: this is a promotion bug, not a data outage → Yes,
  next business day."** This change is that row, made true of one more way a model can be
  unusable. Widen the row's wording; do not add a row.
- **Rule 2** (liberal about missing, strict about malformed): `selected_features` naming a feature
  that does not exist is malformed input at a contract boundary, rejected there. An *absent*
  `selected_features` key is missing rather than malformed, which is why it passes.
- **Rules 6 and 7** are untouched: this adds **no asset check**, so nothing warns and no warning
  path gains a way to raise. Confirmed by tracing the one check that reads the promoted model,
  `_read_promoted_model_facts` (`defs/checks.py:739-775`) — it parses `meta.json` by hand, degrades
  to `_UNKNOWN_PROMOTED_MODEL`, never touches `ParsedFeatures`, and `live_forecasts_are_healthy`
  runs its whole body under a `BaseException` catch-all regardless.
- **Rule 8** (capability in the training loop, not the serving path): partially in tension, and
  worth stating plainly rather than glossing. The check *does* run on the serving path — every
  6-hourly tick calls `load_forecaster_from_dir`. That is deliberate: parsing ~24 strings is cheap
  enough that one function can serve both the promotion gate and the serving path, which is better
  than two implementations that can drift apart. `live_forecasts` gains no branch and no fallback;
  the failure it already had simply moves earlier and acquires a message.
- **Rule 10** ("hysteresis on model promotion"): a promotion that refuses rather than half-lands is
  damping in the small.
- Hypotheses: serves
  [H1 — a service that mostly runs itself](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
  and its test **T1.1**, by removing an `our-bug`-category entry from the
  [intervention log](https://openclimatefix.github.io/nged-substation-forecast/live_service/intervention-log/)
  before it can happen. T1.1 predicts ≥90% of entries are `upstream-contract`; the #463 class is
  `our-bug`.
- Design principles: none traded away.
  [Principle 7, strict contracts at every boundary](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#7-strict-contracts-at-every-boundary),
  is strengthened — the promoted-model directory becomes a validated boundary rather than an
  assumed one.

## Tests

Each is stated with the assertion that fails on `main` today. Construction facts below were checked
by running them, not by reading.

**In `packages/ml_core/tests/test_production_helpers.py`** (beside the existing
`load_forecaster_from_dir` round-trip tests):

1. `test_load_forecaster_from_dir_rejects_an_unparseable_feature` — save a real
   `XGBoostForecaster` whose `selected_features` includes `"local_utc_offset"` (the actual stale
   name, so the test doubles as a regression pin), then `load_forecaster_from_dir(tmp_path)`.
   `pytest.raises(ValueError, match="local_utc_offset")`. **On `main` this returns a forecaster
   without complaint**, so the test fails with "DID NOT RAISE".

   Cheap construction, verified: `XGBoostConfig(selected_features={"local_utc_offset", …})` is
   accepted (`selected_features: set[str]`, no validator), and `save()` with zero trained boosters
   writes a usable `meta.json` carrying the full `model_params` — no training needed.

2. `test_load_forecaster_from_dir_accepts_the_current_vocabulary` — the same shape with
   `{"local_utc_offset_minutes", "temperature_2m", "power_lag_24h", "windchill"}`; asserts no raise
   and that `selected_features` round-tripped. Passes on `main` too. It is the negative control
   that stops the previous test being satisfied by a guard that rejects everything — its docstring
   says so, so a reviewer does not mistake it for a test of this change.

3. `test_a_model_without_selected_features_is_not_rejected` — a hand-written `meta.json` holding
   `model_class` and `model_params` without a `selected_features` key must not raise. Pins the
   deliberate missing-versus-malformed asymmetry (rule 2) so a later edit cannot quietly tighten it
   into a fail-closed. Passes on `main`; stated as a contract pin, not as a test of this change.

**In `packages/ml_core/tests/test_base_forecaster.py`** (file-store MLflow, existing fixtures, no
model-library dependency):

1. `test_fetch_model_artifacts_keeps_the_previous_model_when_the_new_one_is_unservable` — save a
   `_FakeForecaster` with a good vocabulary to run A and promote it to `dest`; save one with
   `selected_features={"local_utc_offset"}` to run B; assert `fetch_model_artifacts(B, dest)`
   raises **and** that `dest`'s `meta.json` still holds run A's vocabulary and `promotion.json`
   still names run A. **On `main` `dest` is overwritten**, so all three post-conditions fail. This
   is the test that pins "a rejected promotion leaves the previous champion in place", which is the
   plan's main departure from the issue's proposed placement.

**In `tests/test_promoted_model.py`** (integration marker, real file-store MLflow + Dagster):

1. `test_promoted_model_refuses_a_model_with_an_unparseable_feature` — assert the materialisation
   fails and `production_model_path` was not created.

   **`dagster.materialize` defaults to `raise_on_error=True`**, so after the change the exception
   propagates out of `materialize` and any `assert not result.success` would never run. Write this
   as `pytest.raises(...)` around the `materialize` call, or pass `raise_on_error=False` and assert
   on `result.success`. On `main` the materialisation succeeds either way, so the test fails there.

   `_save_trained_model_to_mlflow` hard-codes `selected_features={"temperature_2m"}`; give it a
   keyword-only `selected_features` parameter defaulting to that, leaving the three existing tests
   untouched.

No test matches the message beyond the offending feature name — asserting on full prose makes the
test a spellchecker.

## Docs to update

- **`docs/live_service/operations.md`, Step 2** — the "What the asset does" list gains the refusal
  as its own numbered step, ahead of the atomic replace, saying what a refusal looks like and that
  the previous champion is untouched. Add one line under Step 1 warning that the candidate table
  lists *every* fold run ever trained, including ones whose feature vocabulary predates a rename.
- **`docs/live_service/operations.md`, Step 3** — item 1 says "Raises if the model has no trained
  time series (re-promote first)". Extend to the new raise.
- **`docs/design-philosophy/inherent-stability.md`** — the failure-modes row "The promoted model is
  empty or unloadable" becomes "empty, unloadable, or built from features this code no longer
  recognises". A few words; do not add a row.
- **`scripts/build_and_verify_image.sh` header** — the "What it gates on, and what it does not"
  section says the container is expected to fail at the NWP lookup, "which runs *after* the model
  has already loaded. That ordering is the proof the model loaded." Still true. Add a sentence that
  a failure *at* model load naming a feature means the baked-in model predates a feature change and
  must be re-promoted from a re-trained run. Re-read the `grep -qi "mlflow"` gate below it while
  editing — that is where the message constraint above comes from.
- **`docs/live_service/aws.md`** (Step 4's cloud twin, the "dying at the lookup means
  `load_forecaster_from_dir` already succeeded" paragraph) — verify it is still accurate. It should
  be, and stronger; change it only if the new failure mode makes the sentence read as exhaustive.
- **`docs/architecture/production-deployment.md`** and **`docs/architecture/ml-orchestration.md`**
  — both name `fetch_model_artifacts`. Re-read the surrounding paragraphs and adjust only where the
  added validation makes a sentence untrue.
- **Not a roadmap-completing change**: no "Implementation details" deletion, no status-banner edit.
  Confirm against `docs/roadmap/live-service.md` during implementation.

All doc prose in the present tense, describing the code as it then stands — no "used to", no issue
numbers in the prose (CLAUDE.md, "Write about the present, not the past"). Two conventions that
landed on `main` after this plan was first written and apply to the docstring edits above: prose
must be concrete and plain, cut by whole sentences rather than by clipping words (CLAUDE.md, "Prose
style"), and any link from code to a docs page is spelled as its rendered site URL, never as a
`docs/...` path — `base_forecaster.py` and `forecaster.py` were converted wholesale, so match what
is now around you.

## Verification commands

```bash
uv run ruff check . && uv run ruff format . && uv run --all-packages ty check && uv run pytest
```

Docs are touched, so also:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

```bash
uv run mkdocs build --strict
```

No new or changed links are planned, so reading the rendered HTML is needed only if implementation
introduces one. Run the directly-affected suites by hand once:

```bash
uv run pytest tests/test_promoted_model.py packages/ml_core/tests/test_production_helpers.py packages/ml_core/tests/test_base_forecaster.py -v
```

No `--run-network` tests are involved. **Do not run a real promotion against
`data/production_model/` while verifying** — the model there is the current champion.

## Open questions

1. **Should the guard also prove the model *loads* — `model_class` importable, boosters readable —
   at promotion time?** *Recommendation: no.* It costs a full booster load (several GB at V2), needs
   a model library inside `ml_core`'s tests, and the failure is already loud at the first tick.
   Worth its own issue if a promotion ever lands a model that parses but will not load.
2. **Should the booster `feature_names` check go into `XGBoostForecaster.load` anyway?** Three
   lines, no new ABC surface, and it guards the hand-edit-`meta.json` shortcut specifically.
   *Recommendation: no, not in this issue* — already loud at `predict` (verified above), and it
   widens the diff past scope.
3. **Should the guard be widened from "features parse" to "features engineer"?** Parsing catches a
   renamed or removed feature name, not a feature whose name survived but whose engineering changed
   (a newly-required NWP variable, say). The full version is a synthetic-data `engineer()` +
   `predict()` smoke test at promotion time. *Recommendation: no* — much larger, needs a realistic
   `AllFeatures` fixture, and parsing covers every failure this project has actually had.
4. **Should `live_forecasts` degrade rather than raise on an unservable model?** *Recommendation:
   no*, and this is settled rather than open: rule 1 and #446's `trained_ids` argument both put a
   promotion bug on the loud side. Recorded so the choice is visible in review.

## What the adversarial review changed

A fresh sub-agent reviewed this plan with no access to the reasoning behind it. Findings accepted
and folded in above:

- **Routing `fetch_model_artifacts` through `load_forecaster_from_dir` would break two existing
  tests** — `_FakeForecaster` writes `model_class: "fake"`, which `import_class` rejects. Verified.
  This reshaped the whole mechanism: the check now reads `meta.json` directly, which also removes
  the V2 booster-loading cost the first draft had listed as an open risk.
- **`dagster.materialize` defaults to `raise_on_error=True`**, so the planned
  `assert not result.success` would never execute. Verified against the signature.
- **The two stale MLflow runs are already unpromotable** (old `model/` directory layout, rejected by
  `_download_and_unpack_model`). Verified by listing their artifacts. This demolished the first
  draft's headline "the hazard is live right now" and forced the honest forward-looking framing in
  the Verdict.
- **The plan claimed the container smoke test "closes" the code-changed-after-promotion direction.**
  It does not: the script's only automated gate is the MLflow-hermeticity grep, and its own header
  says the model-load-then-NWP ordering is confirmed by eye. Table rewritten; the overclaim is gone.
- **The new error message must not contain "mlflow"** in any casing, or the smoke test hard-fails
  the image. Genuinely non-obvious; now recorded as a constraint on the message.
- **The rule-8 justification was wrong** — the check does run on the serving path, every tick.
  Rewritten to say so and defend it, rather than deny it.
- **`docs/architecture/production-deployment.md`'s "the Docker build reuses this same asset" was
  stale.** Out of scope here, so it was raised separately and has since landed on `main` in
  `64dde4aa`. Nothing left to do.

Findings noted but **not** acted on:

- *"Row 2 of the placement table over-claims: `live_forecasts` already fails today, so detection is
  unchanged."* Half right, and the table now says "fails at load rather than deep in `engineer()`"
  instead of "caught". Rejected as a reason to drop the call site: moving a failure ahead of a
  15-day power read and a full NWP partition scan, and attaching a message that names the remedy, is
  the difference between an operator reading a stack trace and an operator reading an instruction.
- *"`docs/live_service/aws.md`'s twin paragraph should be updated alongside the script header."* The
  paragraph stays true after the change, so it is listed as a check-during-implementation rather than
  a scheduled edit — rewriting prose that is already correct is churn.
