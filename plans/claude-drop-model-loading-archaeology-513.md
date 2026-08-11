# Plan: Drop backwards-compatibility archaeology from the model-loading error paths (#513)

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/513>
Branch: `claude/drop-model-loading-archaeology-513`

All line numbers below are against `main` as merged into this branch on 2026-08-11, which includes
[#486](https://github.com/openclimatefix/nged-substation-forecast/issues/486) (feature-vocabulary
guard), [#512](https://github.com/openclimatefix/nged-substation-forecast/issues/512),
[#508](https://github.com/openclimatefix/nged-substation-forecast/issues/508) and
[#509](https://github.com/openclimatefix/nged-substation-forecast/issues/509).

## Verdict

**Worth implementing, roughly as described.** Both docstrings and both error messages describe a
migration away from a state that exists nowhere: this project is greenfield, has no external users,
and holds no trained model predating either contract. That is the
"[write about the present, not the past](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/)"
rule applied to code comments, and `code-style.md` sanctions the removal directly: comments may be
removed when they are "misleading or out of date".

**The change is five hunks in three files, and the diff is net shorter.** Three hunks are pure
deletions; two replace a sentence with a shorter one, so the diff does contain some new prose —
read those two hunks as rewrites, not deletions. It is worth doing now because both messages are
read while something is broken, and a sentence about a code version that never existed is the worst
thing to read at that moment.

### Departures from the issue body

1. **Both guards stay.** The issue offers deleting the `model_class`-missing branch "and letting
   the `KeyError` speak". Rejected — reasoning below.
2. **The "Out of scope" note is stale.** It defers to
   [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228), which is
   **closed** — the class-resolution swap has landed, and `load_forecaster_from_dir` already
   resolves via `contracts.config_schemas.import_class`.

## The per-branch decision

### `_production_helpers.load_forecaster_from_dir` — keep the guard, strip the archaeology

**What causes it.** In production, a `meta.json` that exists but lacks `model_class` has one cause:
a `BaseForecaster` subclass whose `save` does not honour the contract at `base_forecaster.py:235`.
The directory is unpacked by `fetch_model_artifacts` from an archive built by `save_to_mlflow` from
a subclass's own `save`; a torn `shutil.move` leaves the directory absent or the JSON truncated,
landing in the `FileNotFoundError` branch or a `JSONDecodeError`, never in a well-formed
`meta.json` missing one field. `XGBoostForecaster.save` always stamps it
(`forecaster.py:221`) and is the only non-test subclass, so the causing agent is always a *future*
subclass author.

**Two readers.** The developer who caused it, and an operator who did not:
`inherent-stability.md:146` lists "the promoted model is empty or unloadable" as a hard failure
with "Human alerted? Yes, next business day". An operator who cannot have caused the fault still
needs an action.

**Keep it.** `KeyError: 'model_class'` names neither `meta_path` nor an action. The guard is also
symmetric with the `FileNotFoundError` four lines above — same shape, same job, including its
remedy — and with `_check_meta_features_are_parseable`, the third guard in the same function, which
#486 gave its own remedy string. Stripping the remedy from one of three would make it the odd one
out.

Rewrite the `Raises:` entry and the message to drop the claim about which code version wrote the
file, keeping one remedy and the existing pointer to the contract:

> `{meta_path}` has no 'model_class' field, so the forecaster class cannot be reconstructed.
> Re-promote `promoted_model` from a different run (see BaseForecaster.save).

**"From a different run", singular.** `PromotedModelConfig.mlflow_run_id`
(`production_assets.py:114`) takes an arbitrary run id, so rolling back to the previous champion
*is* re-promoting from a different run — offering them as two options would pad a message read
while production is down. Re-promoting the *same* run id would re-download the same broken
`meta.json`, which is why the message must not say simply "re-promote".

**Implementation constraint:** this message can reach the container log that
`scripts/build_and_verify_image.sh:114` greps case-insensitively for `mlflow` to prove the runtime
is hermetic. The word must not appear in it — the same trap `_production_helpers.py:155-157` already
warns about for its neighbour. Say "run", never "MLflow run".

### `base_forecaster._download_and_unpack_model` — keep the guard, delete a whole sentence

Keep it. The re-raise earns its place because MLflow's raw exception says only that the artifact
path was not found. But the cause is **not** an interrupted training run: `trained_cv_model` creates
the fold run *after* training succeeds (`cv_assets.py:409` trains, `:422` creates, `:424` saves), so
a crashed fold leaves no run at all. The live cause is that `get_or_create_fold_run` **creates** the
run when no tagged one exists, and `cv_power_forecasts` calls it itself at `cv_assets.py:518` then
`load_from_mlflow` at `:520` — so materialising `cv_power_forecasts` for a fold whose
`trained_cv_model` has not run creates an empty fold run and fails on it. `deps=["trained_cv_model"]`
is a lineage edge, not a runtime interlock. That makes the surviving remedy exactly right.

The message currently reads "has no `model.tar.gz` artifact. Either no model was ever saved to this
run, or it was saved before the model became a single archive artifact — re-materialise
`trained_cv_model` for this fold to rewrite it."

**Delete the whole middle sentence, not just its second clause.** A run holds exactly one model
artifact — that is the entire point of `_MLFLOW_MODEL_ARTIFACT` (`base_forecaster.py:21-35`: "A run
holding exactly *one* model file is what the fix rests on") — so "has no `model.tar.gz` artifact"
and "no model was ever saved to this run" are the same statement. Deleting the sentence removes the
archaeology *and* the restatement, and sidesteps the dangling `Either` that trimming only the second
clause would leave.

Also drop the trailing **"to rewrite it"**: nothing pins it, and "rewrite" presupposes an archive
already present in a superseded format — the same historical premise this issue exists to delete.

Result: `"MLflow run {run_id} has no {_MLFLOW_MODEL_ARTIFACT} artifact — re-materialise
\`trained_cv_model\` for this fold."` The remedy is verbatim what `cv_power_forecasts` already gives
at `cv_assets.py:525`.

## What changes, file by file

### `packages/ml_core/src/ml_core/_production_helpers.py`

- `load_forecaster_from_dir`, `Raises:` entry for `ValueError` (lines 212-215). **#486 appended a
  second, unrelated cause to this same entry**, so this is a partial rewrite, not a substring
  deletion — dropping the archaeology alone would leave a dangling em dash and strip the entry's
  remedy, making it the only one of the three without one. Target text, with #486's clause intact:

  ```text
  ValueError: ``meta.json`` has no ``model_class`` field, so the forecaster class cannot be
      reconstructed — re-promote ``promoted_model`` from a different run. Or the model's
      ``selected_features`` name a feature this code cannot parse, in which case re-train
      and promote the new run.
  ```

- The `ValueError` message (lines 226-230): rewrite to the draft above. This is a rewrite of the
  second sentence, not a substring deletion — "Re-promote the model with a code version that stamps
  model_class" becomes "Re-promote `promoted_model` from a different run", and "the *concrete*
  forecaster class" loses "concrete". `(see BaseForecaster.save)` stays.

No change to control flow, exception type, or the `meta.get(...)` call.

### `packages/ml_core/src/ml_core/base_forecaster.py`

Untouched by the merge — both hunks are exactly where they were.

- `_download_and_unpack_model`, `Raises:` entry (lines 83-86): reduce to "The run holds no model
  archive — re-materialise `trained_cv_model` for this fold."
- The `MlflowException` message (lines 94-99): delete the middle sentence and "to rewrite it".

No change to control flow or exception type.

### `packages/ml_core/tests/test_base_forecaster.py`

`test_loading_a_run_with_no_archive_says_what_to_do_about_it` docstring (line 235): **delete** the
sentence "The case that matters is a run written before the model became a single archive
artifact", and do not replace it. The summary line already carries the why ("fails with an
actionable message, not MLflow's raw one"), and the following sentence about MLflow's raw error
being unactionable survives.

Explaining the empty-run mechanism here instead would copy an argument that lives in
`cv_assets.py:518-520` into a test docstring in a *different package*, where nothing can detect
drift — squarely the "one home per argument" rule in `code-style.md`. It would also be easy to get
subtly wrong: `cv_metrics` (`cv_assets.py:916`) resolves a fold run but never calls
`load_from_mlflow`, so it can leave an empty run behind but cannot raise this error.

The `pytest.raises(..., match="re-materialise \`trained_cv_model\`")` assertion at line 243 is
untouched.

## Design-philosophy check

A prose-only diff moves no line: no control flow, no exception type, no asset check, no degradation
path. The one rule that bears on it is the fail-fast line, and it is unchanged —
`load_forecaster_from_dir` raises inside the production `live_forecasts` asset, which
`inherent-stability.md:146` explicitly sanctions ("the promoted model is empty or unloadable | hard
failure | this is a promotion bug, not a data outage"). No principle is traded away and no
engineering hypothesis is in play.

## Tests

**No new test, and no assertion change.** The change is behaviour-preserving by construction, so
**nothing in this diff fails on `main` today** — correct for a prose-only change rather than a gap
to paper over. A test pinning the new wording would be a string literal copied from its own subject.

The three existing tests are a stronger net than that admission suggests. `KeyError` is not a
subclass of `ValueError`, so `test_load_forecaster_from_dir_raises_on_missing_model_class` already
fails if the guard is deleted — a behavioural pin on the guard's *existence*. A repo-wide grep
confirms these are the only three `pytest.raises(..., match=...)` sites touching either message
(`tests/test_promoted_model.py:99,109` assert the *value* of `meta["model_class"]`, not error text):

- `test_load_forecaster_from_dir_raises_on_missing_model_class` (`ValueError`, `match="model_class"`)
- `test_load_forecaster_from_dir_raises_on_missing_dir` (`FileNotFoundError`, `match="Materialise"`)
- `test_loading_a_run_with_no_archive_says_what_to_do_about_it`
  (`MlflowException`, `match="re-materialise \`trained_cv_model\`"`)

The third is the one that could break by accident: its `match` string sits inside the message being
edited. Backticks are not regex metacharacters, so it matches literally — the deleted sentence and
the deleted "to rewrite it" both sit outside the matched substring.

## Docs to update

**None.** No page or docstring quotes either message verbatim. Checked post-merge:
`docs/live_service/operations.md:119-124` and `:317-320`,
`docs/design-philosophy/inherent-stability.md:146-147`,
`docs/architecture/ml-orchestration.md`, `docs/live_service/aws.md`, and
`src/nged_substation_forecast/defs/checks.py:739-775` (which reads the same `meta.json` and
*degrades* rather than raising — correct for an asset check, and independent of this change).

This issue completes no roadmap item, so there is no ship-time triage.

## The sweep

Swept the merged repo (`*.py`, `*.md`, `*.yaml`) for error messages, comments and docstrings
describing migration away from a state the repo no longer has. Most matches for "backwards",
"historical" and "predates" are unrelated domain language. **The merge introduced no new
archaeology**: re-running the sweep after merging #486, #508, #509 and #512 returns the same four
sites.

**In scope, fixed here:** the two sites the issue names (`_production_helpers.py:213`,
`base_forecaster.py:85` and `:97`) plus one it does not, `tests/test_base_forecaster.py:235`.

**Outside `packages/ml_core/` — listed for Jack, not edited.** All three are comments in *tests*,
and all fall on the keep side of this rule: a runtime error path telling its reader to migrate
describes a world that does not exist, whereas a regression test's comment naming the bug it
prevents explains why the test is worth keeping — information not derivable from its assertions.

- `packages/contracts/tests/test_project_root.py:3-7` — "PROJECT_ROOT used to be
  `Path(__file__).parents[4]` … (issue #287)". Without it, a future reader may "simplify" it back.
- `tests/test_trained_cv_model.py:272` — "the exact input change that used to be rejected".
- `tests/test_assets.py:444` and `:446` — "that it no longer rejects such a slice is pinned by …"
  and "What fails on `main` is the count", this repo's own test-writing convention rather than
  archaeology.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check && uv run pytest
```

Fast feedback on the two affected modules:

```bash
uv run pytest packages/ml_core/tests/test_production_helpers.py packages/ml_core/tests/test_base_forecaster.py
```

Because the whole diff is inside docstrings, the markdown lint that matters is the **docstring**
one — `.pre-commit-config.yaml:74-78` runs `scripts/lint_docstring_markdown.py` as a hook separate
from the `docs/` one, and CLAUDE.md's `pymarkdown scan` command covers only `docs/`, READMEs and
CLAUDE.md, none of which this branch touches:

```bash
uv run python scripts/lint_docstring_markdown.py packages/ml_core/src/ml_core/_production_helpers.py packages/ml_core/src/ml_core/base_forecaster.py packages/ml_core/tests/test_base_forecaster.py
```

No network-gated tests are relevant, and no links change.

## Risks and open questions

1. **Keep the `model_class` guard, or delete it?** The alternative — `meta["model_class"]` and let
   the `KeyError` fly — saves ~4 source lines plus a 6-line test. Both simplicity reviews pressed
   it; the second found the repo's own counter-precedent at `_mlflow_runs.py:54`, which resolves the
   identical class-identity contract with a bare `tags["forecaster_target"]` and no guard. That is
   the **R&D** path, where fail-fast-with-a-traceback is the documented policy, so the asymmetry is
   defensible rather than inconsistent. *Recommendation: keep.* If you prefer deletion, the honest
   replacement is a test asserting every `BaseForecaster` subclass stamps `model_class` — its own
   issue.

2. **`_download_and_unpack_model`'s remedy is wrong for one of its two callers.** *(Found in
   review; out of scope, for your call.)* It is reached from `load_from_mlflow` (CV folds) **and**
   from `fetch_model_artifacts` (`_production_helpers.py:271`), which `promoted_model` calls with an
   operator-typed `mlflow_run_id`. An operator who mistypes a run id, or pastes a parent/summary run
   id, is told to "re-materialise `trained_cv_model` for this fold". Fixing it means either dropping
   the remedy (making the message true for both callers — `cv_power_forecasts` already prints the
   same remedy four lines later at `cv_assets.py:525`, so the CV reader is not left empty-handed) or
   branching the message by caller. Either changes the test's `match` string. *Recommendation: not
   in this issue — it is a different defect from archaeology. Worth its own issue; I can file it.*

3. **`promoted_model` will swap in a model it cannot load.** *(Rearchitecture proposed in review;
   your call.)* #486 made `_check_meta_features_are_parseable` run on **both** the promotion path
   (before the atomic swap, `_production_helpers.py:273`) and the serving path (`:234`). The
   `model_class` guard runs on the **serving path only**. So promotion will replace a working
   champion with a directory that cannot be loaded at all, and production breaks at the next tick —
   while the *narrower* fault, a stale feature name, is caught before the swap and the champion
   keeps serving. `inherent-stability.md:147` promises pre-swap refusal for the feature case; row
   146 promises nothing for the unloadable case.
   - **Buys:** one servability function (fold the `model_class` branch into
     `_check_meta_features_are_parseable`, renamed); `load_forecaster_from_dir` shrinks to
     exists-check, parse, one servability call, `import_class`, `load`.
   - **Costs:** `_production_helpers.py`, `test_production_helpers.py`, a case in
     `test_base_forecaster.py`, plus `operations.md` and a row in `inherent-stability.md`. ~40 lines.
     No Delta table, no saved model, no asset signature.
   - **Gives up:** nothing in the current design, but it *is* a behaviour change — promotion starts
     rejecting where today it succeeds and serving fails later.
   - **Now?** *No.* #513 is a prose fix; this is five times larger and changes what
     `inherent-stability.md:146` promises. *Recommendation: file as its own issue.*

4. **Stamping `model_class` in the base class.** *(Rearchitecture proposed in review; recommend
   against.)* Make `BaseForecaster.save` concrete, delegating to an abstract `_save`, then stamp
   `class_target(self)` into `meta.json` itself — the guard becomes unreachable and can be deleted.
   - **Buys:** deletion of a four-line guard and its test; one line out of
     `XGBoostForecaster.save`.
   - **Costs:** `base_forecaster.py`, `xgboost_forecaster/forecaster.py`, three test classes,
     `operations.md`, `docs/roadmap/metrics-and-leaderboard.md:233`. Half a day.
   - **Gives up:** it moves the filename `meta.json` from convention into the base class's
     implementation, so a future forecaster whose natural format is a single file must still emit a
     `meta.json` for the base class to patch. That trades against design principle 5, "everything
     around the model is general-purpose" — and it does not even close the hole, only narrowing
     "write `meta.json` with this field" to "write `meta.json`".
   - **Now?** *No, and I would not do it at all on this evidence.*

## What the reviews changed

Three reviews ran: simplicity, correctness, then simplicity again after merging `main` under the
updated brief that lets a reviewer propose a different architecture.

**Review 1 (simplicity)** cut a speculative clause from the new message, dropped a proposed test
assertion entirely, trimmed the sweep, and caught that the plan's markdown-lint command scans
`docs/` rather than docstrings.

**Review 2 (correctness)** found the fold-run causal story backwards (`trained_cv_model` creates the
run *after* training) and the remedy decision wrong — both rewritten above. Its clean checks:
the three `pytest.raises` sites were the only ones repo-wide, and no doc page quotes either message.

**Review 3 (simplicity, post-merge)** turned two of the plan's own fixes into deletions: the
`base_forecaster` message loses a whole redundant sentence rather than a repaired clause, and the
test docstring loses its archaeology with no replacement — the replacement would have copied an
argument across a package boundary, and was itself imprecise about `cv_metrics`. It also caught
that "re-promote or roll back" were one action written as two, that the plan's claim to track the
`operations.md` runbook was wrong (that passage is about a *different* raise — no trained time
series), and it raised open questions 2, 3 and 4.

**Review 4 (correctness, post-merge)** confirmed the substantive reasoning against merged main —
the redundant-sentence deletion (nothing but `save_to_mlflow` logs an artifact to a fold run, so the
two clauses really are one statement), the test-match survival, the hermeticity constraint traced
end to end, and open question 3's asymmetry. It found three precision defects, all fixed above: a
summary claiming "all deletions, no line added" when two hunks are rewrites; a `Raises:`-entry
instruction that would have left a dangling em dash and stripped the entry's only remedy, now given
as target text; and one stale sweep citation (`test_assets.py` moved +243 lines in the merge).

### Findings rejected

- **"Delete the `model_class` guard."** *Rejected on the bottom line in both simplicity reviews,
  accepted on the reasoning.* Surfaced as open question 1 rather than buried; review 3 independently
  reached the same conclusion after tracing every producer of a `meta.json`.
- **"Cut the sweep section."** *Rejected as stated, applied in part.* The evidence table is gone and
  the out-of-scope hits are now a list, but the sweep stays: the issue explicitly asks for it.

Every other finding across the three reviews was accepted.
