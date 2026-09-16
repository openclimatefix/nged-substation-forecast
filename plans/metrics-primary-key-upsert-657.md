# Give `Metrics` a primary key + upsert write path (#657)

**Problem.** `Metrics` (`packages/contracts/src/contracts/ml_schemas.py`) has no `PRIMARY_KEY` and
no uniqueness check, so nothing stops the `forecast_metrics` Delta table from accumulating
duplicate rows for the same (scope, window). `PowerForecast` already carries this pattern —
`PRIMARY_KEY: ClassVar[tuple[str, ...]]` plus a `validate()` override that raises on a duplicated
key — and should be mirrored.

**Solution.** Add the ten-column key the issue specifies to `Metrics`, with a `validate()`
override mirroring `PowerForecast.validate()`'s `n_unique()` check but guarded for the four key
columns that are `allow_missing` (they are not yet populated at the two call sites that validate
before enrichment). Correct the class docstring and the `computed_at` docstring to match the new
key, and correct `docs/roadmap/live-service.md`'s now-inaccurate description of monitoring rows as
append-only-and-disambiguated-by-`computed_at`. No change to the write path
(`write_forecast_metrics` in `delta_store/forecast_metrics.py`) and no write-path design note
added to the roadmap doc: its existing partition-scoped overwrite is already correct for both
current callers (`leaderboard`, `ad_hoc`), which always recompute their entire
`(experiment_name, fold_id)` partition in one call, and the question of what a future
`production_monitoring` writer needs has no caller to build it for yet (`EvalScopeType`
deliberately excludes that scope) — see departure 2 below for why this plan does not try to answer
it in advance.

## Verdict, size, departures

**Worth it, roughly as described.** The current absence of a key is a real gap: nothing stops
`compute_metrics()` fanning out into duplicate rows, and nothing documents what "the same row
computed again" means. **Size: Complex** (changes a data contract) — full plan, both plan reviews,
both diff reviews, per the issue's own sizing call.

Departures from the issue body:

1. **The `validate()` override cannot check the full key unconditionally**, unlike
   `PowerForecast`. Two call sites validate `Metrics` before `enrich_metrics_rows()` adds
   `experiment_name`, `evaluation_scope`, `window_start`, and `window_end` — four of the ten key
   columns, all declared `allow_missing=True` on `Metrics`: inside `compute_metrics()` itself
   (`packages/ml_core/src/ml_core/metrics.py:473`) and in `_score_forecast_group`
   (`src/nged_substation_forecast/defs/cv_assets.py:857`, `Metrics.validate(pl.concat(batch_metrics),
   allow_superfluous_columns=True)`, called on `compute_metrics()`'s already-validated output). At
   both, the four columns do not exist on the frame at all (not merely null) — confirmed
   empirically against Patito's `allow_missing` behaviour — so `validated_df.select(pk_cols)` would
   raise `ColumnNotFoundError` if the check ran unconditionally. `PowerForecast`'s four key columns
   are all mandatory, so its own `validate()` never has to consider this. The override here skips
   the uniqueness check unless every key column is present, which means the real check runs where
   it should: inside `enrich_metrics_rows()`'s own `Metrics.validate()` call
   (`packages/ml_core/src/ml_core/metrics.py:600`), once the full key exists, on exactly the frame
   that gets written. This still catches everything the issue is after (a join fanning out inside
   `compute_metrics()`, or double-writing rows) — it just can't also fire on the two partial-key
   frames upstream, which have no key to check yet.

   *Considered and rejected*: keep `Metrics` self-enforcing by putting the uniqueness check in a
   `validate()` override (as above), versus dropping the override and adding a plain classmethod
   called explicitly from `enrich_metrics_rows()` once the key exists — the latter needs no
   presence guard and drops one test, since there would be nothing to skip. Rejected because it
   moves contract enforcement out of `contracts` and into a caller in `ml_core`, against this
   repo's stated design (`packages/contracts/README.md`: duplicate/uniqueness checks live inside a
   model's own `validate()` and are never relaxed) — and unlike the write-path question in
   departure 2, the presence guard exists for two callers that are real and present today, not a
   hypothetical future one.
2. **No write-path code changes, and no write-path design note added to the roadmap doc.** The
   issue asks the plan to "determine what the write path needs to become" once monitoring writes
   land incrementally — read as asking for a decision, not for that writer to be built now, since
   no caller exists yet to build it for (`EvalScopeType` in ml_schemas.py is deliberately narrower
   than `EVALUATION_SCOPES`, per its own docstring, until Phase 8). An earlier draft of this plan
   recorded a MERGE-vs-predicate-scoped-overwrite recommendation in `docs/roadmap/live-service.md`
   for that future writer; dropped after review, because it rests on no measurement of the real
   table (this environment has no AWS credentials to take one) and would commit a future
   implementer to an unverified choice ahead of the work being agreed — the "don't commit the
   project to work it has not agreed to" rule. Phase 8's implementer re-derives the write strategy
   against real data when that work starts.
3. **`docs/roadmap/live-service.md` needs a correction, not just `ml_schemas.py`.** That page
   currently describes monitoring rows as "**append-only** ... unlike the leaderboard scope's
   idempotent overwrite; recomputations are distinguished by `computed_at`" — i.e. it documents the
   opposite of the issue's model (append + disambiguate, rather than key + replace). Once `Metrics`
   carries a key that includes `window_start`/`window_end`, that sentence becomes factually wrong —
   a recomputed window now collides on the key and replaces rather than appending — so this is a
   direct consequence of the schema change, not scope creep, and the correction stays in this plan
   even though the write-path recommendation (departure 2) was cut.

## What changes, file by file

### `packages/contracts/src/contracts/ml_schemas.py`

- Add imports: `ClassVar`, `Self` from `typing`; `Sequence` from `collections.abc` (mirrors
  `power_schemas.py`).
- `Metrics`: add

  ```python
  PRIMARY_KEY: ClassVar[tuple[str, ...]] = (
      "time_series_id",
      "power_fcst_model_name",
      "experiment_name",
      "fold_id",
      "evaluation_scope",
      "horizon_slice",
      "metric_name",
      "metric_param",
      "window_start",
      "window_end",
  )
  ```

  with a one-line docstring naming the grain: at most one metric value per series, model,
  experiment, fold, scope, horizon slice, metric, parameter, and window.
- Add a `validate()` override, signature identical to `PowerForecast.validate()`. Body:
  call `super().validate(...)`, then only run the uniqueness check when
  `set(cls.PRIMARY_KEY).issubset(validated_df.columns)` — skip it otherwise (the partial-key call
  site). When it runs, use `validated_df.select(pk_cols).n_unique() != validated_df.height`, not
  `is_duplicated()`, for the same reason `PowerForecast` gives (materialising a per-row mask costs
  ~5x peak memory). Docstring explains both the raise-not-degrade rationale (own bug, not the
  outside world misbehaving — copy `PowerForecast`'s framing and link) and the presence guard
  (points at `enrich_metrics_rows()` as where the real check fires).
- Rewrite the class docstring's stated primary key (currently six columns) to the ten-column key
  above, and update the worked-example table's caption to match.
- Rewrite `computed_at`'s description: it no longer "orders the append-only monitoring series and
  distinguishes recomputations" — recomputing an existing key replaces that row, so `computed_at`
  is provenance (when this row was last computed) and nothing else. State that explicitly so a
  reader does not go looking for code that uses it to disambiguate.

### `docs/roadmap/live-service.md`

In "The `production_monitoring` evaluation scope" (around line 296): replace "These rows are
**append-only** — successive runs accumulate the sliding-window history (unlike the leaderboard
scope's idempotent overwrite; recomputations are distinguished by `computed_at`)" with a
description matching the new model — each run's `(window_start, window_end)` is normally distinct
from the last (a trailing window is calculated relative to "now"), so most runs do append a new
row, but a run that recomputes an existing `(window_start, window_end)` — a retried sensor firing,
a backfill — replaces that row rather than duplicating it, keyed on `Metrics.PRIMARY_KEY`. Nothing
else in that section changes: how a future writer implements the replace (a key-scoped predicate,
a `MERGE`, or something else) is left to whoever builds it, since nobody has measured the real
table's write cost yet.

### `packages/delta_store/src/delta_store/forecast_metrics.py`

No change.

## Design-philosophy check

This is R&D-path code (`metrics` asset, tagged `RESEARCH_LAYER_TAGS`), not the production serving
path, so the inherent-stability degrade-don't-raise rule does not apply — R&D is the documented
exception, and raising on a duplicate key here is the same choice `PowerForecast` already made for
the same reason: a duplicated primary key is our own bug (a fanned-out join, a double write), not
the outside world misbehaving, so it should raise rather than silently corrupting a leaderboard or
a monitoring chart. No asset check is added or changed, so the `WARN`/`blocking=False` rule does
not apply here. No `docs/design-philosophy/engineering-hypotheses.md` hypothesis is being tested by
this change.

## Tests

`packages/contracts/tests/test_metrics.py` (new file, mirrors `test_power_forecast.py`):

- `test_metrics_rejects_duplicate_primary_key` — two rows identical on all ten key columns
  (constructed with every `allow_missing` column populated) → `Metrics.validate()` raises
  `ValueError` matching "Duplicate entries found for primary key". **Fails on `main` today**
  because `Metrics` has no `PRIMARY_KEY` and no `validate()` override, so duplicate rows pass
  silently.
- `test_metrics_accepts_rows_differing_only_in_window_end` (or `window_start`) — two otherwise
  identical rows with different `window_end` → validates cleanly. **Passes on `main` today for
  the wrong reason** (no check at all); after the change it passes because the key correctly
  treats different windows as distinct rows.
- `test_metrics_validate_skips_uniqueness_check_when_key_columns_missing` — a frame with only the
  six always-present columns (`time_series_id`, `power_fcst_model_name`, `fold_id`,
  `horizon_slice`, `metric_name`, `metric_param`) plus duplicated rows on those six, validated with
  `allow_missing_columns=True` (or however `Metrics.validate()` is normally called pre-enrichment)
  → does not raise. **This is the regression this plan must not introduce**: without the presence
  guard, this call raises `ColumnNotFoundError` instead of validating, which would break both
  `compute_metrics()` (`ml_core/metrics.py:473`) and `_score_forecast_group`
  (`cv_assets.py:857`) on `main`'s own calling pattern once `PRIMARY_KEY` is added. Reproduce the
  guard's absence by writing this test first against a version of the override with no guard,
  confirming it fails with `ColumnNotFoundError`, then confirming the guarded version passes.

`packages/delta_store/tests/test_forecast_metrics.py`: no new tests needed —
`test_overwrite_is_partition_scoped_by_experiment_and_fold` already exercises the write path and
is untouched by this change (no code changes there).

`tests/test_metrics.py` (root-level integration tests for the `metrics` asset — 13 tests,
including `test_metrics_is_idempotent` and `test_score_forecast_group_per_series_batches`, the two
most relevant to this change): no new tests needed here either, but this file is the regression
check that both partial-key `Metrics.validate()` call sites (`compute_metrics()` and
`_score_forecast_group`) still work once `PRIMARY_KEY` exists — it must stay green unchanged.
`test_metrics_is_idempotent` in particular already exercises re-running the same group and
asserting the row count does not grow, so it would have caught a coarser regression than this
plan's own new tests target.

## Docs to update

- `packages/contracts/src/contracts/ml_schemas.py`: class docstring (primary key, ten columns) and
  `computed_at` field docstring (provenance-only, not disambiguation) — both listed above.
- `docs/roadmap/live-service.md`: "The `production_monitoring` evaluation scope" section —
  append-only wording, listed above.
- `docs/roadmap/metrics-and-leaderboard.md` line 331 references "the primary key includes
  `metric_param`" in an unrelated context (a different discussion, about `metric_param="all"` not
  colliding with new parametric rows) — checked, and it stays correct under the new ten-column key
  without editing; flagging only so the implementer does not need to re-check it.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest packages/contracts/tests/test_metrics.py packages/delta_store/tests/test_forecast_metrics.py -v
uv run pytest  # full suite — cv_assets.py's metrics-asset tests must stay green
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict  # docs/roadmap/live-service.md edit
```

## Risks and open questions

1. **Is `evaluation_scope` really needed in the key, given `fold_id` already distinguishes
   `"live"` (monitoring/ad-hoc) from a real fold label?** The issue's own reasoning (leaderboard
   and monitoring rows would collide without it) only bites once `ad_hoc` and
   `production_monitoring` can both produce `fold_id="live"` rows over the *same* window — which
   they can (an ad-hoc analyst could evaluate the same trailing window an automated monitoring run
   already covered). Recommend keeping it, as the issue specifies.
2. **Confirm the `docs/roadmap/live-service.md` wording change is the intended design change**,
   not an unintended side effect of this issue. The issue body argues for it directly (`computed_at`
   is "pure provenance... under this model recomputations are not distinguished, they replace"),
   but that roadmap page was written describing the opposite behaviour, so this plan is knowingly
   reversing a previously-agreed design point for the as-yet-unbuilt monitoring writer.
