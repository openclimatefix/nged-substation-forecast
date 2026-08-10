# Plan — #486: reject a promoted model the running code cannot serve

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/486>
Branch: `claude/reject-unservable-promoted-model-486`

## Verdict

**Worth implementing, but the issue's account of what the guard buys is wrong, and the guard
belongs one layer lower than the issue puts it.**

The hazard is real and live *right now*. `list_promotable_runs()` on Jack's laptop returns three
candidates, and two of them carry the **pre-#463** feature vocabulary:

| run id | experiment | vocabulary |
|---|---|---|
| `7acafbe3…` | `xgboost_cv_0002` | `local_utc_offset_minutes` (current) |
| `aecc8c3f…` | `xgboost_no_power_lags` | `local_utc_offset` (**stale**) |
| `cf8ddd29…` | `xgboost_cv_0001` | `local_utc_offset` (**stale**) |

`promotable_model_runs` renders all three side by side in one metadata table with nothing to
distinguish them, and the pick is explicitly "by eye" (`operations.md`, Step 1). Two thirds of the
menu is poison. That alone justifies the change.

### Departure 1 — the guard would *not* have caught the incident the issue opens with

The issue frames this as the fix for the #463 breakage. It is not, and the plan should not claim
it is. The timeline, verified from git and from the promoted artefacts on disk:

- PR #463 (the `local_utc_offset` → `local_utc_offset_minutes` rename) merged at
  **2026-08-07 14:22 +0100**.
- The model that was being served at that moment had been promoted **before** the rename.
- The model currently on disk (`data/production_model/promotion.json`) was promoted at
  **2026-08-07 17:35 UTC**, i.e. *after* the rename, and carries the new name.

So #463 was a *promote-then-change-the-code* failure. A guard that runs only at promotion time
runs before the breaking change and sees nothing wrong. What it catches is the **mirror-image**
case — promoting a model whose vocabulary the *current* code cannot parse — which is exactly the
live hazard tabulated above, and exactly the trap an operator falls into when they reach for
`xgboost_cv_0001` off the candidate list.

The plan therefore covers **both directions**, and says which mechanism covers which. That is the
substantive addition to the issue, and it costs one extra call site rather than a second feature.

### Departure 2 — validate inside `load_forecaster_from_dir`, not inside `promoted_model`

The issue proposes validating in the `promoted_model` asset after `fetch_model_artifacts`, and
notes as an open question that validating before the atomic swap would be better. It would, and
there is a placement that gets that *and* covers the second direction with no duplication:

Put the check in **`ml_core._production_helpers.load_forecaster_from_dir`**, and have
`fetch_model_artifacts` call `load_forecaster_from_dir` against the freshly-unpacked staging
directory *before* the swap. One check function, one place it is written, three places it fires:

| Call site | Direction caught | Effect |
|---|---|---|
| `fetch_model_artifacts`, pre-swap | promoting a stale model into current code | Promotion refuses; the previous working champion stays on disk untouched |
| `live_forecasts` (already calls `load_forecaster_from_dir`) | code changed after promotion | The tick fails *at model load*, before the 15-day power read and the NWP scan, with a message naming the feature and saying "retrain and re-promote" |
| `scripts/build_and_verify_image.sh` smoke test | code changed after promotion, before the AWS deploy | The offline container run fails at model load with the same message, ahead of the expected NWP failure |

The third row is the one that closes #463's own direction for the AWS deployment: the image is
built from the *current* checkout but copies `data/production_model/` as promoted earlier, so the
smoke test is the first moment the new code and the old artefact meet. That script's header already
documents that the model loads *before* the NWP lookup fails, so the guard lands inside the window
the script already exercises.

Validating in `promoted_model` itself instead would leave a broken model written to
`production_model_path` (the issue spots this), and would miss both other rows.

### Departure 3 — booster `feature_names` comparison: **rejected**, with evidence

The issue's second open question asks whether to compare `Booster.feature_names` against
`selected_features`. Verified empirically against the installed xgboost 3.4.0:

- `xgb.QuantileDMatrix(polars_df, …)` sets `feature_names` from the frame's columns, and those
  names survive a `.ubj` `save_model`/`load_model` round-trip.
- `Booster.predict` validates them and raises
  `ValueError: feature_names mismatch: [...] [...] / training data did not have the following
  fields: …`.

`XGBoostForecaster._feature_cols` is `sorted(self.model_params.selected_features)` and
`_prepare_features` selects exactly those columns, so a `meta.json`-vs-booster disagreement already
fails loudly, with a message that names the offending column, at the first `predict`. It is not the
*silent* class of failure this issue exists to close. It can also only arise from hand-editing
`meta.json`, which is already ruled out. Adding it would need either a new hook on the
`BaseForecaster` ABC or model-specific code in a model-agnostic helper — real machinery for a
failure mode that is already loud and that nothing has produced. Not in this change.

(If Jack wants it anyway, the cheap version is a self-consistency check inside
`XGBoostForecaster.load` — the boosters are already loaded there, so it costs three lines and no
new ABC surface. Flagged under Open questions.)

## What changes, file by file

### `packages/ml_core/src/ml_core/_production_helpers.py`

- **New private function `_check_selected_features_are_parseable(forecaster, path)`.** Calls
  `ParsedFeatures.from_strings(forecaster.model_params.selected_features)` and, on `ValueError`,
  re-raises a `ValueError` `from` it whose message states: the directory, the model's
  `experiment_name`, the underlying parse error (which already names the offending feature), and
  the remedy — *re-train and re-promote; never hand-edit `meta.json`*. Nothing else; no swallowing.
- **`load_forecaster_from_dir`** — call it on the loaded forecaster before returning, and add a
  `Raises:` entry for the new `ValueError`. Its docstring gains one sentence saying the returned
  forecaster is one this code can actually engineer features for, not merely one it could
  deserialise.
- **`fetch_model_artifacts`** — between `_download_and_unpack_model` and the `rmtree`/`move`,
  call `load_forecaster_from_dir(downloaded_dir)` and discard the result. Ordering: unpack →
  validate → write `promotion.json` → swap, so a rejected promotion writes nothing at all and
  `dest` is never touched. Docstring: state that the atomic swap now also gates on the model being
  loadable *and* servable by this code, and that a rejected promotion leaves the previous champion
  in place.
- **Module docstring** — it currently calls both disk/MLflow helpers "thin, single-purpose IO
  wrappers". Reword to reflect that they are the promotion/serving gate.

Import note: `ml_core.base_forecaster` already imports from `ml_core.features`, so importing
`ParsedFeatures` here adds no new cycle.

### `src/nged_substation_forecast/defs/production_assets.py`

- `promoted_model` — no code change. Its docstring gains a sentence that promotion now refuses a
  model whose `selected_features` this code cannot parse, and that the refusal happens before the
  directory swap so the previous champion survives.
- `live_forecasts` — no code change (it already calls `load_forecaster_from_dir`). Its docstring's
  "Loads the production model…" paragraph gains the same fact, next to the existing empty-model
  raise.

### `packages/ml_core/src/ml_core/features/__init__.py`

Check `ParsedFeatures` is exported; if it is only reachable as
`ml_core.features._parsed_features.ParsedFeatures`, import it from the private module rather than
widening the package's public surface for one caller.

## Design-philosophy check

This code path straddles both postures, and the split is exactly the one
`docs/design-philosophy/inherent-stability.md` already draws.

- **Rule 1** reserves raising for "states that are our own bug — an empty promoted model, a
  contract violation". A promoted model whose feature vocabulary the serving code cannot parse is
  the same class as the empty-promoted-model raise sitting fifteen lines away in `live_forecasts`.
  It is not the outside world misbehaving; it is an artefact we produced meeting code we wrote.
  #486's own comment makes this argument by reference to #446, and #446 makes it in the opposite
  direction for NWP staleness. Consistent.
- The failure-modes table already carries the row **"The promoted model is empty or unloadable →
  Hard failure — the asset raises → Unchanged: this is a promotion bug, not a data outage → Yes,
  next business day."** This change is that row, made true of one more way a model can be
  unloadable. The row's wording is widened rather than a new row added (see Docs below).
- **Rule 2** (liberal about missing, strict about malformed): `selected_features` naming a feature
  that does not exist is malformed input at a contract boundary, rejected there.
- **Rule 6/7** are untouched: this adds **no asset check**, so nothing warns, nothing runs
  `blocking=False`, and no warning path gains a way to raise.
- **Rule 8** (capability in the training loop rather than the serving path): the guard is a
  promotion-time gate, not a serving-path branch; `live_forecasts` gets no new `if`. What it gets
  is the *same* failure it already had, moved earlier and given a message.
- **Rule 10** ("damp the corrections… hysteresis on model promotion"): a promotion that refuses
  rather than half-lands is damping in the small.
- Hypotheses: this serves
  [H1 — a service that mostly runs itself](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
  and its test **T1.1**, by removing an `our-bug` intervention category from the
  [intervention log](https://openclimatefix.github.io/nged-substation-forecast/live_service/intervention-log/)
  before it can happen — T1.1 predicts ≥90% of entries are `upstream-contract`, and the #463 class
  of incident is `our-bug`.
- Design principles: no principle is traded away.
  [Principle 7, strict contracts at every boundary](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#7-strict-contracts-at-every-boundary),
  is *strengthened* — the model directory becomes a validated boundary rather than an assumed one.

## Tests

All new tests are unit tests in `packages/ml_core/tests/test_production_helpers.py`, alongside the
existing `load_forecaster_from_dir` round-trip tests, except where noted. Each is stated with the
assertion that fails on `main` today.

1. **`test_load_forecaster_from_dir_rejects_an_unparseable_feature`** — save a real
   `XGBoostForecaster` whose `selected_features` includes a name the current code cannot parse
   (`"local_utc_offset"` — the actual stale name, so the test doubles as a regression pin), then
   `load_forecaster_from_dir(tmp_path)`. Assert `pytest.raises(ValueError, match=...)` naming the
   offending feature. **On `main` this returns a forecaster happily** — the load path never touches
   `ParsedFeatures` — so the test fails on `main` with "DID NOT RAISE".

   Construction note: `XGBoostConfig(selected_features={"local_utc_offset", …})` is accepted by
   pydantic (`selected_features: set[str]`, no validator), and training is not required — `save()`
   with no boosters still writes a `meta.json` with `model_params` and an empty
   `trained_time_series_ids`. Keep the test that cheap.

2. **`test_load_forecaster_from_dir_accepts_the_current_vocabulary`** — the same shape with
   `{"local_utc_offset_minutes", "temperature_2m", "power_lag_24h", "windchill"}`, asserting no
   raise and that `model_params.selected_features` round-tripped. Passes on `main` too; it is the
   negative control that stops finding #1 by making the guard reject everything. Stated as such in
   its docstring so a reviewer does not mistake it for a test of this change.

3. **`test_fetch_model_artifacts_leaves_the_previous_model_in_place_when_the_new_one_is
   _unservable`** — in `packages/ml_core/tests/test_base_forecaster.py`, where the existing
   `fetch_model_artifacts` tests live (file-store MLflow, `MLFLOW_ALLOW_FILE_STORE=true` per the
   existing fixtures). Save a good model to run A, promote it to `dest`; save a stale-vocabulary
   model to run B; assert `fetch_model_artifacts(B, dest)` raises **and** that `dest`'s
   `meta.json` still names run A's experiment and `promotion.json` still names run A. **On `main`
   this overwrites `dest`**, so both post-conditions fail.

4. **`test_promoted_model_refuses_a_model_with_an_unparseable_feature`** — in
   `tests/test_promoted_model.py` (integration marker, real file-store MLflow + Dagster, reusing
   `_save_trained_model_to_mlflow` with a `selected_features` parameter added). Assert
   `materialize(...)` does **not** succeed and that `production_model_path` was not created.
   **On `main` the materialisation succeeds**, so `assert not result.success` fails. This is the
   test that pins the issue's actual headline claim at the asset level.

   `_save_trained_model_to_mlflow` currently hard-codes `selected_features={"temperature_2m"}`;
   give it a keyword-only `selected_features` parameter defaulting to that, so the three existing
   tests are untouched.

No test asserts on the *exact* wording of the message beyond the offending feature name — matching
full prose makes the test a spellchecker.

## Docs to update

- **`docs/live_service/operations.md`, Step 2** — the numbered "What the asset does" list gains the
  refusal as its own step, before the download's atomic replace, saying what a refusal looks like
  and that the previous champion is untouched. Also worth one line under Step 1 warning that the
  candidate table lists *every* fold run ever trained, including ones whose feature vocabulary
  predates a rename — which is precisely the state Jack's MLflow store is in today.
- **`docs/live_service/operations.md`, Step 3** — the "What the asset does" item 1 currently says
  "Raises if the model has no trained time series (re-promote first)". Extend to the new raise.
- **`docs/design-philosophy/inherent-stability.md`** — the failure-modes row "The promoted model is
  empty or unloadable" becomes "empty, unloadable, or built from features this code no longer
  recognises". One-word-scale edit; do not add a row.
- **`docs/architecture/production-deployment.md:353`** — describes `fetch_model_artifacts` as "a
  pure, asset-independent helper". Read the surrounding paragraph and adjust if the new validation
  makes the sentence untrue.
- **`docs/architecture/ml-orchestration.md:55`** — mentions `fetch_model_artifacts` unpacking into a
  temporary directory. Check whether the pre-swap validation belongs in that sentence.
- **`scripts/build_and_verify_image.sh` header** — the "What it gates on, and what it does not"
  section says the container is expected to fail at the NWP lookup, "which runs *after* the model
  has already loaded. That ordering is the proof the model loaded." Still true, and now stronger:
  add a sentence that a failure *at* model load with the feature-vocabulary message means the
  baked-in model predates a feature change and must be re-promoted from a re-trained run.
- **Not a roadmap-completing change**, so no "Implementation details" deletion and no status-banner
  edit. Confirm against `docs/roadmap/live-service.md` during implementation.

Everything above is written in the present tense, describing how the code works now — no "used to",
no issue numbers in the prose (CLAUDE.md, "Write about the present, not the past").

## Verification commands

The green-before-push set:

```bash
uv run ruff check . && uv run ruff format . && uv run --all-packages ty check && uv run pytest
```

Docs were touched, so also:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

```bash
uv run mkdocs build --strict
```

No new or changed links are planned, so reading the rendered HTML is only needed if implementation
introduces one. Specific to this change, and worth running by hand once:

```bash
uv run pytest tests/test_promoted_model.py packages/ml_core/tests/test_production_helpers.py packages/ml_core/tests/test_base_forecaster.py -v
```

No `--run-network` tests are involved. **Do not** run a real promotion against `data/production_model/`
while verifying — the model on disk is the current champion.

## Risks and open questions

1. **Does `fetch_model_artifacts` loading the model cost too much at V2 scale?** At ~2,500 series
   the boosters are several GB. *Recommendation: accept it.* `live_forecasts` already loads every
   booster on every 6-hourly tick, so promotion loading them proves nothing more than that the
   serving path can. If it does not fit at promotion, it does not fit at serving either, and
   finding that out with an operator present is the point.

2. **Should the booster `feature_names` check go in `XGBoostForecaster.load` anyway?** Three lines,
   no new ABC surface, and it guards exactly the hand-edit-`meta.json` shortcut. *Recommendation:
   no, not in this issue* — it is already loud at `predict` (verified above), and it widens the
   diff past the issue's scope. Easy to add later as its own issue if the temptation to hand-edit
   ever bites.

3. **Should the guard be widened from "features parse" to "features engineer"?** Parsing catches a
   renamed or removed feature name. It does not catch a feature whose *name* survived but whose
   engineering changed (a new required NWP variable, say). The full version is a synthetic-data
   smoke `engineer()` + `predict()` at promotion time. *Recommendation: no.* That is a much larger
   change needing a realistic `AllFeatures` fixture, and the parse check covers every failure this
   project has actually had. Worth its own issue if a second incident argues for it.

4. **Should `live_forecasts` degrade rather than raise on an unservable model?** *Recommendation:
   no* — and this is settled, not open: rule 1 and #446's `trained_ids` argument both put a
   promotion bug on the loud side. Recorded here only so the choice is visible in review.
