# Plan: make `live_forecasts` independent of the metadata roster (#528)

**The problem.** `live_forecasts` reads the `TimeSeriesMetadata` roster through
`_load_engineering_inputs` (`cv_assets.py:371`) on every 6-hourly slot, and that read is the only
source of two things the live path cannot do without: the H3 cells the NWP scan is pruned to
(`cv_assets.py:379`), and the static per-series columns the feature engineer joins on
(`tabular_feature_engineer.py:229-230`). An unreadable roster therefore raises inside a production
asset, and a roster missing rows for series the model was trained on silently drops those series
from the forecast. Both breach [rule
1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules):
never raise in production because an input is absent or stale.

**The solution.** Freeze the static per-series metadata into the saved model artifact, which is the
issue's own preferred direction and the one [rule
8](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#where-complexity-should-live)
points at. `BaseForecaster.save_to_mlflow` deposits a `time_series_metadata.parquet` into the model
directory before archiving it, so it rides inside the existing single `model.tar.gz` and lands in
`Settings.production_model_path` when `fetch_model_artifacts` unpacks it. `live_forecasts` reads
that file instead of the roster. `_load_engineering_inputs` stops reading the roster at all: the
roster read becomes its own function, `_load_roster`, which only the R&D callers call — so the
fail-fast posture stays exactly where it is today and no production code path touches the roster.

## Verdict

**Worth implementing, in the issue's own preferred direction (candidate 1).** The failure it
describes is reachable and its cost is total: NGED gets nothing every six hours until a human
intervenes.

Two corrections to the issue body, both verified against `main` at 66de9c6c:

- The issue says "#508 makes the hourly ingest rebuild an unusable roster rather than failing on
  it, which shortens the exposure to at most one hourly run". **That is not what shipped.**
  `assets.py:135-147` wraps `upsert_metadata` in a `try`/`except` that logs, calls
  `report_asset_degradation` and records `metadata_upsert_failed`. It never rewrites the roster, so
  `upsert_metadata`'s `TimeSeriesMetadata.validate(existing_metadata)` (`storage.py:418`) raises
  again on every subsequent hourly run. **Exposure to an unusable roster is unbounded, not one
  hour**, which makes #528 more valuable than its body claims, not less.
- The issue treats a *thin* roster as a live-path failure mode of equal standing. It is reachable
  but only via roster loss: `upsert_metadata` takes the union of the stored roster and the new
  snapshot (`storage.py:424`, `:449`) and its docstring is explicit that "a series that
  `new_metadata` omits keeps its last stored values indefinitely", so the roster never shrinks in
  normal operation. It goes thin when the file is deleted or replaced — which is exactly the
  operator's recovery action for the *unusable* case above — and is then recreated from one NGED
  snapshot. So the two failure modes are the same incident at two stages, and one fix should close
  both. It does: after this change the live path reads no roster, so neither an unusable nor a thin
  roster can affect a forecast.

### Departures from the issue body

- **Candidate 3 (Delta time travel on the roster) is dropped.** It depends on #533, which is OPEN,
  is not a sub-issue of this epic, and is a substantial job in its own right (its body documents
  five measured `deltalake` traps). Planning on top of unlanded work is not on.
- **Candidate 2 (split `_load_engineering_inputs` into a production and an R&D entry point) is
  adopted as a consequence, not as the fix.** Splitting alone changes no behaviour: production
  would still have nothing to forecast from when the roster is unreadable, which is the issue's own
  objection to "wrap the read in a `try`". Candidate 1 makes the split fall out for free, because
  the roster read has exactly one caller left.
- **The metadata does not become state on the forecaster object.** The issue phrases candidate 1 as
  "the promoted model … carr[ying] the static per-series features". It is carried by the model's
  saved *directory*, not by an attribute on `BaseForecaster`: nothing about fitting or predicting
  uses it, so putting it through `train`, an abstract property and each subclass's `save`/`load`
  would plumb state through the ML interface to no end. `base_forecaster.py:254-255` already
  sanctions this shape — "Depositing a file *after* a save is fine, and is how
  `_production_helpers.fetch_model_artifacts` puts `promotion.json` beside the model" — and
  `checks.py:816-852` already reads a model-directory file with a plain function. Consequence:
  **no code in `packages/xgboost_forecaster/` changes** (one docstring line does), and models saved
  before this change still `load` — so `cv_power_forecasts` replays of old fold runs are
  unaffected.
- **The `forecasts.height == 0` raise at `production_assets.py:285` stays.** Once metadata travels
  with the model, the only remaining input that can empty the forecast is NWP — and "NWP absent, or
  too old to cover the horizon" is
  [#446](https://github.com/openclimatefix/nged-substation-forecast/issues/446)'s scope, listed
  as such in the failure-modes table of `inherent-stability.md`. Removing the raise here would
  fix half of #446 in the wrong issue and leave the other half (widened bands, a degradation
  marker on the row) undone.
- **No Patito contract changes.** What is persisted is a subset of the existing
  `TimeSeriesMetadata` rows, one `allow_missing` column lighter (see below), so it still validates
  against the contract and no field is added, widened or relaxed in `packages/contracts/`.

## What changes, file by file

### `packages/ml_core/src/ml_core/base_forecaster.py`

- New module constant `TRAINED_METADATA_FILENAME: Final[str] = "time_series_metadata.parquet"`,
  beside `_MLFLOW_MODEL_ARTIFACT`, with a docstring saying why the file exists and why it is inside
  the archive rather than a second MLflow artifact (a second artifact would reopen the merge
  problem `_MLFLOW_MODEL_ARTIFACT`'s docstring documents).
- `save_to_mlflow` gains a keyword-only `time_series_metadata: pt.DataFrame[TimeSeriesMetadata]`,
  written into `model_dir` **after** `self.save(model_dir)` — `save` clears the directory, so the
  order matters — and before `_archive_model_dir`. Required, not optional: a caller that trains a
  model has the roster in hand, and a model uploaded without it cannot be promoted.
  It writes through `write_trained_metadata`, which drops `area_wkt`; see below.
- Two new module functions defining the file's layout in one place:
  `write_trained_metadata(model_dir, metadata)`, which does the `area_wkt` drop and the write, and
  `load_trained_metadata(model_dir) -> pt.DataFrame[TimeSeriesMetadata]`. `save_to_mlflow` calls
  the first; `live_forecasts`, `fetch_model_artifacts` and the test fixtures that hand-build a
  production model directory call one or the other, so no caller re-states the filename or the
  drop.
  `load_trained_metadata` uses `set_model`, not `validate`, mirroring how `_load_engineering_inputs`
  reads the roster today — this is load-bearing, not stylistic: `TimeSeriesMetadata.validate` fails
  with eight missing-column errors on the three-column frames `tests/test_live_forecasts.py`'s
  fixtures write, so validating here would break every test in that file.
  It raises a `ValueError` naming the remedy (re-promote) when the file is absent — a model
  directory missing it is a promotion bug, the same class as "the promoted model is empty or
  unloadable", which the failure-modes table already lists as a deliberate hard failure, and the
  same class as `load_forecaster_from_dir`'s existing raise on a missing `meta.json`.
  Both live here rather than in `_production_helpers` because that module imports *from* this one
  (`_production_helpers.py:28`), and this module owns the model-directory layout
  (`_archive_model_dir`).

**Why `area_wkt` is dropped, measured on the real V1 roster** (`data/NGED/metadata.parquet`, 32
rows, 12 columns): the frame is 129,582 bytes in memory and `area_wkt` is **127,635 of them —
98.5%**. Every other column together is under 2 KB. `area_wkt` is `allow_missing=True`, nothing in
the feature pipeline reads it, and at V2 scale (~2,500 series) it would put megabytes of polygon
text into every fold's `model.tar.gz`, for every experiment, forever. Dropping only that one column
keeps the frame validatable against `TimeSeriesMetadata` and keeps `latitude`/`longitude` (128
bytes each) available for the rung-4 clear-sky floor when it is built.

### `packages/ml_core/src/ml_core/_production_helpers.py`

- `fetch_model_artifacts` refuses a staged model whose `time_series_metadata.parquet` is absent,
  unreadable, or does not cover the model's trained population, immediately after
  `_check_meta_is_servable` (`:286`) and **before** the `rmtree`/`move` swap (`:294-297`). That is
  the existing refuse-before-the-swap pattern, and it turns a would-be 06:00 production failure
  into a promotion that is declined while the outgoing champion keeps serving. It reads the staged
  file through `load_trained_metadata`, so a corrupt file is refused on the same path as a missing
  one. The population comes from `meta.get("trained_time_series_ids", [])` — **not** a subscript:
  `BaseForecaster.save` mandates only `model_class` in `meta.json` (`base_forecaster.py:238-256`),
  `trained_time_series_ids` is an `XGBoostForecaster` convention (`forecaster.py:221`), and every
  other reader treats it as optional (`checks.py:829`, `production_assets.py:155`). A subscript
  would turn a future subclass into a bare `KeyError` with none of the "which run, what to do"
  wording the neighbouring refusals carry. The docstring's `Raises:` section gains the case.

### `src/nged_substation_forecast/defs/cv_assets.py`

- New `_load_roster(settings, time_series_ids) -> pt.DataFrame[TimeSeriesMetadata]`, holding
  exactly the `pl.read_parquet` + filter + `set_model` that is inline at `:371-375` today. Its
  docstring says it is an R&D-only entry point and names the reason: production gets its metadata
  from the promoted model directory.
- `_load_engineering_inputs` takes `metadata: pt.DataFrame[TimeSeriesMetadata]` as an argument and
  returns `tuple[pt.LazyFrame[PowerTimeSeries], pt.LazyFrame[Nwp]]`. The `cells` derivation at
  `:378` now reads from the passed frame. Everything else in the function is unchanged, including
  all three NWP pruning levers and their docstring.
- `trained_cv_model` (`:448`) calls `_load_roster`, then `_require_metadata_coverage`, then
  `_load_engineering_inputs`, and passes the roster to `forecaster.save_to_mlflow(...)` at `:475`.
- `cv_power_forecasts` (`:591`) calls `_load_roster` **once before** the `init_time` chunk loop and
  `_require_metadata_coverage` on it, instead of re-reading the roster on every chunk. The
  `is_first` guard around `_require_metadata_coverage` disappears with it. Its comment
  (`:600-601`) gives two reasons — "metadata does not vary by init_time window, and raising on a
  later chunk would leave the partition holding a partial fold" — and calling it once *before* the
  loop honours both, because it still raises before the first `write_power_forecasts`.
- `forecast_metrics` (`:1103`) is untouched: it reads the roster directly, not through
  `_load_engineering_inputs`, and R&D should keep failing fast on it.

### `src/nged_substation_forecast/defs/production_assets.py`

- `live_forecasts` gets `metadata_df = load_trained_metadata(Path(settings.production_model_path))`
  filtered to `trained_ids`, and passes it to both `_load_engineering_inputs` and
  `feature_engineer.engineer`. No roster read remains in the production path.
- **The read goes immediately before the `_load_engineering_inputs` call — after
  `_available_nwp_init_times` and `select_nwp_init_time`, not before them.** `docs/live_service/aws.md:640-643`
  makes the current ordering a runbook step: the offline smoke test and the first cloud run both
  prove the model loaded by dying at `_available_nwp_init_times` with `TableNotFoundError`, "so
  dying at the lookup means `load_forecaster_from_dir` already succeeded". Reading the metadata
  earlier would change what a healthy smoke test looks like and invalidate that text. A comment
  says so, so a later tidy-up does not move it.
- The `power_time_series_and_metadata` entry in `deps` stays — the asset still reads that asset's
  power Delta.
- The asset docstring gains a short paragraph: static per-series features come from the model
  directory, so the roster's state cannot fail or thin a live slot, and the H3 cells the NWP scan
  is pruned to are the cells the model trained against rather than whatever the roster says today.

### `scripts/build_and_verify_image.sh`

Promotion is gated by `fetch_model_artifacts`, but the **image bake bypasses promotion entirely**:
`Dockerfile:61` does `COPY data/production_model/ data/production_model/`, and the script's only
automated gate is the `grep -qi mlflow` hermeticity check (a non-zero container exit is *expected*
in the offline smoke test). So an image baked from a directory promoted before this change would
build, print `[PASS]`, deploy, and then fail every 6-hourly slot. Add the same four-line hard fail
the script already has for a missing `data/production_model/promotion.json` (`:61-67`), for
`data/production_model/time_series_metadata.parquet`, pointing at re-promotion. This is the gate
that makes the hard raise in `load_trained_metadata` safe to ship.

## Design-philosophy check

- **Production degrades, R&D fails fast.** After this change the production path's only inputs are
  the model directory, the NWP Delta and the power Delta. The roster is read by `trained_cv_model`,
  `cv_power_forecasts` and `forecast_metrics` — all R&D, all still raising through
  `_require_metadata_coverage` and the bare read. Rule 1 is satisfied by removing the dependency
  rather than by catching an exception, which the issue rightly rejects as "failing quietly".
- **Rule 8, complexity into the training loop.** The capability — knowing where each series is —
  moves from a serving-time read to a training-time record. This is the rule's central example.
- **Rule 4, signal degradation in-band.** Nothing new to signal: the degradation this removes is no
  longer reachable. `live_forecasts_are_healthy` already reports any trained series a slot did not
  forecast (`checks.py:980`, comparing against the promoted model's `trained_time_series_ids`), and
  it needs no change.
- **Rule 6/7, checks warn and cannot raise.** No asset check is added or edited.
- **Design principle 1, the power forecast never stops** — this is the principle the issue is
  filed against, and the change delivers it for the roster input specifically.
- **What the feature pipeline actually consumes today**, so the docstrings claim no more than is
  true: `time_series_id` (the semi-join at `tabular_feature_engineer.py:170`), `h3_res_5` (the
  spatial join at `:59-62`) and — only when `"time_series_type"` is in `selected_features` — that
  one column (`:229-230`). `latitude` and `longitude` are in the persisted frame but **no code
  reads them yet**: `StaticFeature` is `Literal["windchill"]` and its registry entry is a weather
  formula (`_parsed_features.py:18-27`).
- **Hypotheses.** This serves `H1` (the forecast is produced in every input state) by removing one
  state in which it is not. It does not move any rung of the ladder by itself: rung 4's clear-sky
  floor is not built, and this change makes the `latitude` that floor will need available in
  production — a precondition for that work, not the work.
- **Train==predict, extended.** A side effect worth stating: today a roster edit between training
  and serving silently changes a live series' `h3_res_5` (so its weather cell) and its static
  feature values, while the model was fitted on the old ones. Freezing the metadata into the
  artifact makes the static half of the train==predict invariant hold the same way the population
  half already does.

## Tests

Each new assertion, and why it fails on `main` today.

**`tests/test_live_forecasts.py`**

- `test_a_trained_series_losing_its_metadata_row_does_not_stop_the_others` is **rewritten**, not
  deleted — its premise is exactly what this issue changes. The rewritten test writes a roster
  covering series 1 and 3, saves a model carrying metadata for both, materialises and asserts
  `{1, 3}`; then removes series 3 from the roster, then overwrites the roster with junk bytes, then
  deletes it outright, asserting after each step that the next slot still forecasts `{1, 3}`. All
  three steps fail on `main`: the thin roster yields `{1}`, and the corrupt and missing files raise
  out of `pl.read_parquet` inside `_load_engineering_inputs`. Deleting the file is the strictly
  strongest assertion (it proves nothing even `stat()`s the path), but the thin step is the issue's
  second named failure mode and the one that fails *silently*, and the corrupt step is the
  incident #508 was filed for.
- **Both** model-directory fixtures deposit a `time_series_metadata.parquet` via
  `write_trained_metadata`, as `fetch_model_artifacts` does in production: `_save_promoted_model`
  (`:129`), which the `env` fixture uses for every test in the file, and `_save_model_trained_on`
  (`:324`), which the degradation test uses. Missing either one fails the whole file.
- New `test_live_forecasts_refuses_a_model_directory_with_no_metadata`: delete the file from the
  production model directory and assert the materialisation fails naming the remedy (via
  `materialize(..., raise_on_error=False)`, the pattern already used at
  `tests/test_trained_cv_model.py:316`). This is the deliberate hard failure — a promotion bug, not
  an input outage — and pins it so a later change cannot quietly turn it into an empty forecast.

**`tests/test_trained_cv_model.py`**

- The two existing `test_load_engineering_inputs_*` tests (`:171`, `:192`) are updated for the new
  signature — they build metadata through `_load_roster` and pass it in. Signature updates, not new
  coverage.
- The existing end-to-end training materialisation gains assertions rather than getting a sibling
  test: unpack the fold run's archive and assert `load_trained_metadata` returns rows covering
  `trained_time_series_ids` with the roster's `h3_res_5`, and no `area_wkt` column. Fails on
  `main`: the archive holds no such file. As a separate test it would re-run the whole slow
  training materialisation for one assertion, since pytest tests share no state.

**`packages/ml_core/tests/test_base_forecaster.py`**

- A fifth case in the existing `test_fetch_model_artifacts_keeps_the_previous_model_when_the_new_one_is_unservable`
  parametrised table (`:305-345`): `_save_without_trained_metadata`, which archives a `save()`d
  directory without the deposit (the pre-change archive shape — i.e. what an old fold run actually
  holds). The table already asserts the champion in `dest` survives, which is the property that
  matters. This replaces a Dagster-level promotion test in `tests/test_promoted_model.py`: the
  machinery for building a defective archive lives here, and an integration test would only
  re-prove what this case proves.
- `test_fetch_model_artifacts_unpacks_the_archive_into_dest` (`:264`) asserts the exact directory
  listing at `:243-250`; `time_series_metadata.parquet` joins it.
- New `test_load_trained_metadata_raises_on_a_directory_without_it`, asserting the message names
  re-promotion. Fails on `main`: the function does not exist.
- New `test_save_to_mlflow_carries_the_trained_metadata_without_area_wkt`: save with a metadata
  frame that has an `area_wkt` column, unpack, assert the file is there with the same rows and no
  `area_wkt`. Deliberately **not** a dtype-preservation test — that `pl.Enum` and `Float32` survive
  a parquet round-trip is Polars' behaviour, verified once by hand, not ours to pin.
- The `save_to_mlflow` call sites gain the argument: `:156`, `:289`, `:298`, `:307` here, **and
  `tests/test_promoted_model.py:81`**, which the first draft of this plan missed.

## Docs to update

- `docs/design-philosophy/inherent-stability.md` — the failure-modes table (`:136-151`) has no
  roster row. Add one: "The metadata roster is unreadable, or missing rows for trained series" →
  Today: live inference does not read it; static per-series features are frozen into the model
  archive. Ingest contains the upsert failure and records `metadata_upsert_failed`. → Intended:
  unchanged. → Human alerted: no.
- `docs/live_service/operations.md:232-240` — "Reading a failed roster upsert" currently leaves the
  reader to infer what a stalled roster costs the forecast. State it: nothing, for the live slots,
  because inference reads the model's own frozen copy; what is lost is the metadata change itself,
  and the next training run is where it matters.
- `docs/architecture/production-deployment.md:298` — the paragraph on the model directory should
  name `time_series_metadata.parquet` among its contents and say the live path has no roster
  dependency.
- `docs/architecture/ml-orchestration.md:53-75` — the "one archive file" section should note that
  the trained metadata rides inside the same archive, so the one-artifact property is unchanged.
- `docs/ml_experimentation/dagster-workflow.md:124` quotes the call as
  `forecaster.save_to_mlflow(fold_run_id)`, which the new required argument falsifies.
- `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py:234` — `load`'s docstring
  enumerates the files a model directory can hold that the model did not write
  ("`fetch_model_artifacts` adds a `promotion.json`"); it now also holds the metadata parquet.
  This is the only line in `packages/xgboost_forecaster/` that changes, and it is prose.
- `packages/xgboost_forecaster/README.md` needs **no** change: the file is written by
  `save_to_mlflow`, not by `XGBoostForecaster.save`, so the subclass's saved-directory listing is
  still complete and correct.
- `docs/live_service/aws.md:640-643` needs **no** change, because the plan keeps the metadata read
  after the NWP-availability lookup precisely so that text stays true.
- No roadmap item completes here, so no ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format . && uv run --all-packages ty check && uv run pytest
```

Plus, because docs change:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict
```

No network-gated tests are involved; nothing here touches NWP conversion conventions.

## Risks and open questions

1. **The currently promoted champion (`xgboost_cv_0002`, promoted 2026-08-07) has no
   `time_series_metadata.parquet`, so `live_forecasts` will refuse to run against it.** Under
   CLAUDE.md's "a change that invalidates an existing trained model costs us a retrain, not a
   migration path" this is acceptable, and it is caught by machine rather than by a note: the
   `fetch_model_artifacts` refusal blocks a promotion that would break it, and the new
   `build_and_verify_image.sh` gate blocks an image bake from a stale directory. The deploy
   sequence is re-train, re-promote, rebuild the image — stated in the PR body. **Jack's call** if
   he would rather keep the current champion serving and defer the whole issue.
2. **`cv_power_forecasts` still forecasts against the roster, not the model's frozen copy.** Making
   it use `load_trained_metadata` would make CV and live agree exactly, which is the stronger
   position for leaderboard comparability. *Recommendation:* not in this issue — it changes what
   R&D reads, where the issue is about production, and R&D wants to fail fast on a roster problem
   rather than silently score against a months-old snapshot. Worth its own issue if the
   train==predict argument above persuades. **Jack's call.**
3. **`production_assets.py:44` imports `_load_engineering_inputs`, a private helper, from the
   research module `cv_assets.py`.** After the `_load_roster` split that function is roster-free
   and purely "load power and NWP for a window", so nothing research-specific remains in it and the
   private production→research import becomes gratuitous. Moving it to `ml_core._production_helpers`
   or a shared `defs/_engineering_inputs.py` is a natural follow-up. Flagging, not fixing — out of
   scope for #528.
4. **`power_data_is_fresh` derives its expected population from the roster** (`checks.py:387`), so
   a thin roster still narrows what that check watches. Out of scope: it is a warning path, it
   already handles an absent roster (`object_exists` guard at `:293`), and its job is to report on
   ingested data rather than on the model's population. Flagging, not fixing.

## Second review: what it changed, and what was rejected

The correctness review confirmed the plan's account of `main` (including the `area_wkt`
measurement, re-run independently) and found six real defects, all now fixed above: the
`meta["trained_time_series_ids"]` subscript, three missed call sites
(`tests/test_promoted_model.py:81`, the exact-directory-listing assertion at
`packages/ml_core/tests/test_base_forecaster.py:243-250`, and the `_save_promoted_model` fixture
the `env` fixture actually uses), two tests placed where they cost far more than they buy, the
ungated image-bake path, and the read-ordering that `docs/live_service/aws.md:640-643` depends on.

Nothing it raised was rejected. Two of its "not a defect" observations were acted on anyway,
because both improve the plan: the round-trip test was reframed so it stops pinning Polars'
behaviour as if it were ours, and the `is_first` comment is now quoted in full so a reviewer can
see the constraint it states is respected.

## First review: findings rejected, and why

The simplicity review's remaining proposals that did not make it in:

- **Snapshot the roster at promotion time (inside `fetch_model_artifacts`) rather than at training
  time.** Verified viable — promotion runs on the laptop against the same local roster, and the
  Docker build COPYs the resulting directory. Rejected because the snapshot would then be the
  roster as of *promotion*, not as of *training*, which is exactly the train==predict gap this
  change is meant to close, for a saving of roughly one function.
- **Persist only `time_series_id`, `h3_res_5` and `time_series_type`.** Rejected in favour of
  dropping `area_wkt` alone: the measurement above shows that one column is 98.5% of the frame, so
  trimming further saves under 2 KB, costs the ability to validate the frame against
  `TimeSeriesMetadata`, and throws away the `latitude` the rung-4 floor will need.
- **Drop the corrupt-roster test and keep only the missing-roster one.** Accepted in substance —
  they are now two steps of one test rather than two tests — but not dropped entirely: a future
  reader that used the repo's own `object_exists` pattern (`checks.py:293`) would tolerate absence
  and still raise on corruption, so the cheaper of the two assertions is worth its two lines.
