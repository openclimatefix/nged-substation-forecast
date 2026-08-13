# Plan: make `live_forecasts` independent of the metadata roster (#528)

**The problem.** `live_forecasts` reads the `TimeSeriesMetadata` roster through
`_load_engineering_inputs` (`cv_assets.py:371`) on every 6-hourly slot, and that read is the only
source of two things the live path cannot do without: the H3 cells the NWP scan is pruned to
(`cv_assets.py:379`), and the static per-series columns the feature engineer joins on
(`tabular_feature_engineer.py:230`). An unreadable roster therefore raises inside a production
asset, and a roster missing rows for series the model was trained on silently drops those series
from the forecast. Both breach [rule
1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules):
never raise in production because an input is absent or stale.

**The solution.** Move the static per-series metadata into the trained model, which is the issue's
own preferred direction and the one [rule
8](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#where-complexity-should-live)
points at. `train` gains the metadata frame it was engineered against, `save` writes it into the
model directory, `load` reads it back, and `live_forecasts` hands
`forecaster.trained_time_series_metadata` to the feature engineer instead of reading
`metadata.parquet`. `_load_engineering_inputs` stops reading the roster at all: the roster read
becomes its own function, `_load_roster`, which only the R&D callers call — so the fail-fast
posture stays exactly where it is today and no production code path touches the roster.

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
- **The `forecasts.height == 0` raise at `production_assets.py:285` stays.** Once metadata travels
  with the model, the only remaining input that can empty the forecast is NWP — and "NWP absent, or
  too old to cover the horizon" is
  [#446](https://github.com/openclimatefix/nged-substation-forecast/issues/446)'s scope, listed
  as such in the failure-modes table of `inherent-stability.md`. Removing the raise here would
  fix half of #446 in the wrong issue and leave the other half (widened bands, a degradation
  marker on the row) undone.
- **No Patito contract changes.** What the model persists is a subset of the existing
  `TimeSeriesMetadata` rows, so no field is added, widened or relaxed anywhere in
  `packages/contracts/`.

## What changes, file by file

### `packages/ml_core/src/ml_core/base_forecaster.py`

- `BaseForecaster.train` gains a third argument, `time_series_metadata:
  pt.DataFrame[TimeSeriesMetadata]` — the same frame the caller handed
  `feature_engineer.engineer`. Its docstring says why: a model's static per-series features are
  part of what it trained on, so they belong to the model, not to whatever the roster says at
  serving time.
- A new abstract property `trained_time_series_metadata -> pt.DataFrame[TimeSeriesMetadata]`,
  written to mirror `trained_time_series_ids`: the model's own frozen record of the static features
  for its trained population, persisted and reconstructed through the subclass's `save`/`load`.
  Its docstring states the two invariants callers rely on — it covers exactly
  `trained_time_series_ids`, and it is the *only* source of `h3_res_5`, `latitude` and `longitude`
  the production path has.
- `save`'s docstring gains a third bullet to its "requirements on every implementation" list: the
  saved directory must also carry the trained metadata.

### `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`

- `__init__` initialises `self._trained_metadata: pt.DataFrame[TimeSeriesMetadata] | None = None`.
- `train` stores the caller's frame filtered to `self.trained_time_series_ids` — after fitting, so
  a requested series that produced no Booster is not carried.
- `trained_time_series_metadata` returns it, raising if `train`/`load` never set it (a programming
  error, not an input failure).
- `save` writes `time_series_metadata.parquet` into the model directory, after the `shutil.rmtree`
  clear. Parquet rather than a `meta.json` field because it round-trips `TimeSeriesMetadata`'s four
  `pl.Enum` columns and its `Float32` lat/lon exactly, where JSON would need a re-cast on the way
  back in. The file rides inside the single `model.tar.gz` that `save_to_mlflow` builds, so it does
  **not** reopen the multi-artifact merge problem `_MLFLOW_MODEL_ARTIFACT`'s docstring warns about.
- `load` reads it back with `pt.DataFrame(pl.read_parquet(...)).set_model(TimeSeriesMetadata)` —
  `set_model`, not `validate`, matching how `_load_engineering_inputs` reads the roster today, so
  this change does not smuggle in new strictness about roster shape.

Constant for the filename lives next to the class as a `Final`.

### `src/nged_substation_forecast/defs/cv_assets.py`

- New `_load_roster(settings, time_series_ids) -> pt.DataFrame[TimeSeriesMetadata]`, holding
  exactly the `pl.read_parquet` + filter + `set_model` that is inline at `:371-375` today. Its
  docstring says it is an R&D-only entry point and names the reason: production gets its metadata
  from the promoted model.
- `_load_engineering_inputs` takes `metadata: pt.DataFrame[TimeSeriesMetadata]` as an argument and
  returns `tuple[pt.LazyFrame[PowerTimeSeries], pt.LazyFrame[Nwp]]`. The `cells` derivation at
  `:379` now reads from the passed frame. Everything else in the function is unchanged, including
  all three NWP pruning levers and their docstring.
- `trained_cv_model` (`:448`) calls `_load_roster`, then `_require_metadata_coverage`, then
  `_load_engineering_inputs`, then passes the roster to `forecaster.train(...)`.
- `cv_power_forecasts` (`:591`) calls `_load_roster` **once before** the `init_time` chunk loop and
  `_require_metadata_coverage` on it, instead of re-reading the roster on every chunk. The
  `is_first` guard around `_require_metadata_coverage` disappears with it — its own comment says it
  exists only because "metadata does not vary by init_time window".
- `forecast_metrics` (`:1103`) is untouched: it reads the roster directly, not through
  `_load_engineering_inputs`, and R&D should keep failing fast on it.

### `src/nged_substation_forecast/defs/production_assets.py`

- `live_forecasts` takes `metadata_df = forecaster.trained_time_series_metadata` from the loaded
  model and passes it to both `_load_engineering_inputs` and `feature_engineer.engineer`. No
  roster read remains in the production path.
- The `power_time_series_and_metadata` entry in `deps` stays — the asset still reads that asset's
  power Delta.
- The asset docstring gains a short paragraph: static per-series features come from the model, so
  the roster's state cannot fail or thin a live slot, and the H3 cells the NWP scan is pruned to
  are the cells the model trained against rather than whatever the roster says today.

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
- **Hypotheses.** This serves `H1` (the forecast is produced in every input state) by removing one
  state in which it is not. It does not by itself move any rung of the ladder: rung 4's clear-sky
  floor still is not built, and this change makes `latitude` available to it in production, which
  is a precondition for that work rather than the work itself.
- **Train==predict, extended.** A side effect worth stating: today a roster edit between training
  and serving silently changes a live series' `h3_res_5` (so its weather cell) and its static
  feature values, while the model was fitted on the old ones. Carrying the metadata makes the
  static half of the train==predict invariant hold the same way the population half already does.

## Tests

Each new assertion, and why it fails on `main` today.

**`tests/test_live_forecasts.py`**

- `test_a_trained_series_losing_its_metadata_row_does_not_stop_the_others` is **rewritten**, not
  deleted — its premise is exactly what this issue changes. The rewritten test writes a roster
  covering series 1 and 3, trains and saves a model on both, materialises and asserts `{1, 3}`,
  then removes series 3 from the roster and asserts the next slot still forecasts `{1, 3}`. On
  `main` the second materialisation yields `{1}`. (The test's current name and docstring go with
  it; the new name says the roster no longer decides the live population.)
- New `test_live_forecasts_survives_an_unreadable_roster`: after the healthy materialisation,
  overwrite `metadata.parquet` with junk bytes and assert the next slot still succeeds and still
  writes the same series. On `main` this raises out of `pl.read_parquet` inside
  `_load_engineering_inputs` and the run fails.
- New `test_live_forecasts_survives_a_missing_roster`: delete `metadata.parquet` entirely and
  assert the slot succeeds. Distinct from the above because a missing file and a corrupt file take
  different paths through Polars' object-store reader, and the missing case is what an operator
  produces while recovering from the corrupt one.
- `_save_model_trained_on` (`:306`) and the `_write_metadata` fixture (`:119`) grow a metadata
  frame handed to `train`, so the saved model carries one.

**`tests/test_trained_cv_model.py`**

- The two existing `test_load_engineering_inputs_*` tests (`:171`, `:192`) are updated for the new
  signature — they now build their metadata through `_load_roster` and pass it in. These are
  signature updates, not new coverage.
- New `test_trained_cv_model_saves_the_metadata_it_trained_on`: after the existing end-to-end
  training materialisation, load the model from its fold run and assert
  `trained_time_series_metadata`'s `time_series_id` set equals `trained_time_series_ids` and that
  its `h3_res_5` matches the roster's. Fails on `main`: the property does not exist.

**`packages/xgboost_forecaster/tests/test_forecaster.py`**

- New `test_save_load_round_trips_trained_metadata`: train on two series, save, load from the
  directory, assert the frame comes back with the same rows, the same column set and the same
  dtypes (specifically that the `pl.Enum` columns and `Float32` lat/lon survive). Fails on `main`:
  nothing is written.
- New `test_train_records_metadata_only_for_series_that_got_a_booster`: request three series where
  one has no non-null `power` in the window, and assert `trained_time_series_metadata` covers the
  two trained ones. Fails on `main` for the same reason.

**`packages/ml_core/tests/test_base_forecaster.py`**

- The `_FakeForecaster` gains the new property and the `train` argument; its existing round-trip
  tests then also cover the base-class contract. Add
  `test_trained_time_series_metadata_is_abstract`, mirroring
  `test_trained_time_series_ids_is_abstract` (`:116`): a subclass omitting it cannot be
  instantiated. Fails on `main`: there is no abstract member to omit.

## Docs to update

- `docs/design-philosophy/inherent-stability.md` — the failure-modes table has no roster row. Add
  one: "The metadata roster is unreadable, or missing rows for trained series" → Today: live
  inference does not read it; static per-series features travel with the promoted model. Ingest
  contains the upsert failure and records `metadata_upsert_failed`. → Intended: unchanged. → Human
  alerted: no.
- `docs/live_service/operations.md:232-240` — "Reading a failed roster upsert" currently leaves the
  reader to infer what a stalled roster costs the forecast. State it: nothing, for the live slots,
  because inference reads the model's own copy; what is lost is the metadata change itself, and
  the next training run is where it matters.
- `docs/architecture/production-deployment.md:298` — the paragraph on the model directory should
  name `time_series_metadata.parquet` among its contents and say the live path has no roster
  dependency.
- `docs/architecture/ml-orchestration.md:53-75` — the "one archive file" section should note that
  the trained metadata rides inside the same archive, so the one-artifact property is unchanged.
- `packages/xgboost_forecaster/README.md:15-16` — the saved-directory listing gains
  `time_series_metadata.parquet`.
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

1. **The saved metadata carries every `TimeSeriesMetadata` column, including `area_wkt`.** At V1
   (32 series) that is kilobytes. At V2 (~2,500 series) the WKT polygons could add megabytes to
   every fold's `model.tar.gz`. *Recommendation:* store the frame as received and leave it. Pruning
   columns would mean deciding which static columns a future feature set might want, which is
   guessing; if the archive size becomes a problem it is a one-line `select` later, and #533's
   measurements suggest we should measure before trimming.
2. **The base class trusts each subclass's `save`/`load` to persist the new state.** Same fragility
   as the existing "two requirements on every implementation" contract in `save`'s docstring, and
   one subclass exists today. The alternative is to make `train`/`save`/`load` concrete template
   methods on `BaseForecaster` wrapping abstract `_fit`/`_save`/`_load`, so the base owns the
   metadata's persistence and no future forecaster can forget it. *Recommendation:* not now — it
   renames three abstract methods to buy an invariant for a second forecaster that does not exist
   yet. Worth doing when the `nged_incumbent` baseline (#147) lands and there are two.
3. **A model promoted before this change has no `time_series_metadata.parquet`, so `load` raises.**
   Under CLAUDE.md's "a change that invalidates an existing trained model costs us a retrain, not
   a migration path" this is acceptable, and the currently promoted `xgboost_cv_0002` would need
   re-training and re-promoting. *Recommendation:* accept, and say so in the PR body so the
   promotion is not a surprise. **Jack's call** if he would rather keep the current champion
   serving.
4. **`cv_power_forecasts` still forecasts against the roster, not the model's metadata.** Making it
   use `trained_time_series_metadata` would make CV and live agree exactly, which is the stronger
   position for leaderboard comparability. *Recommendation:* not in this issue — it changes what
   R&D reads, where the issue is about production, and R&D wants to fail fast on a roster problem
   rather than silently score against a months-old snapshot. Worth its own issue if the
   train==predict argument in the design-philosophy section persuades. **Jack's call.**
5. **`power_data_is_fresh` derives its expected population from the roster** (`checks.py:387`), so
   a thin roster still narrows what that check watches. Out of scope here — it is a warning path,
   it already handles an absent roster (`object_exists` guard at `:293`), and the check's job is to
   report on ingested data rather than on the model's population. Flagging it, not fixing it.
