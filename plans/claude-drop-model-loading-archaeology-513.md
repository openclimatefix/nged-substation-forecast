# Plan: Drop backwards-compatibility archaeology from the model-loading error paths (#513)

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/513>
Branch: `claude/drop-model-loading-archaeology-513`

All line numbers are against `main` as merged into this branch on 2026-08-12.

## Verdict

**Worth implementing, but half of it is already done.** Since this plan was first written, `main`
rewrote `_production_helpers.load_forecaster_from_dir` — the `model_class` guard moved into a new
`_check_meta_is_servable` helper, and **its archaeology went with it**. The remaining work is three
hunks in two files.

The surviving prose describes a migration away from a state that exists nowhere: this project is
greenfield, has no external users, and holds no trained model predating the archive contract. That
is the
"[write about the present, not the past](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/)"
rule applied to code comments, and `code-style.md` sanctions the removal directly — comments may be
removed when they are "misleading or out of date".

### What `main` already fixed

`_production_helpers.py:135-193` now holds `_check_meta_is_servable`, which validates
`model_class`, the config against `CONFIG_CLASS`, and the feature vocabulary, and returns the
forecaster class. Three consequences for this plan:

- **The `model_class` archaeology is gone.** The guard's message (`:166-171`) now reads "The model
  at {source} has no 'model_class' field, so the concrete forecaster class cannot be reconstructed
  (see BaseForecaster.save)", plus a shared `remedy` string. No claim about which code version
  wrote the file. Nothing left to do.
- **The guard was kept**, not deleted — which settles this plan's open question 1 in the direction
  it recommended.
- **The promotion/serving asymmetry is fixed**, which was open question 3. `fetch_model_artifacts`
  applies the same check before the swap, and `load_forecaster_from_dir`'s docstring now says so
  outright: "Promotion applies the same check, so this fires only when the code changed after the
  champion was promoted."

The hermeticity constraint this plan flagged is now written into the code as a comment at
`_production_helpers.py:159-161`, so a future editor cannot miss it.

### Departures from the issue body

1. **The `_production_helpers` half of the issue is complete.** The issue names it as one of two
   sites; only the other one survives.
2. **The `base_forecaster` guard stays.** The issue floats deleting a guard "and letting the
   `KeyError` speak"; that option only ever applied to the `model_class` branch, which is now moot.
3. **The "Out of scope" note is stale.** It defers to
   [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228), which is closed.

## The decision: `base_forecaster._download_and_unpack_model`

Keep the guard, delete a whole sentence. The re-raise earns its place because MLflow's raw
exception says only that the artifact path was not found.

The cause is **not** an interrupted training run: `trained_cv_model` creates the fold run *after*
training succeeds (`cv_assets.py` trains, then creates, then saves), so a crashed fold leaves no run
at all. Two live causes remain, and both are covered by tests on `main`:

- `get_or_create_fold_run` **creates** the run when no tagged one exists, and `cv_power_forecasts`
  calls it itself and then `load_from_mlflow` on the next line — so materialising
  `cv_power_forecasts` for a fold whose `trained_cv_model` has not run creates an empty fold run
  and fails on it. `deps=["trained_cv_model"]` is a lineage edge, not a runtime interlock.
- An operator passing `promoted_model` a mistyped or stale run id that names a real run which never
  held a model — pinned by `test_base_forecaster.py:350-369`, whose docstring calls it "the
  likeliest operator slip of the three".

The message currently reads: "has no `model.tar.gz` artifact. Either no model was ever saved to
this run, or it was saved before the model became a single archive artifact — re-materialise
`trained_cv_model` for this fold to rewrite it."

**Delete the whole middle sentence, not just its second clause.** A run holds exactly one model
artifact — that is the point of `_MLFLOW_MODEL_ARTIFACT` (`base_forecaster.py:21-35`: "A run holding
exactly *one* model file is what the fix rests on"), and `save_to_mlflow` is the only artifact-
logging call in the repo — so "has no `model.tar.gz` artifact" and "no model was ever saved to this
run" are the same statement. Deleting the sentence removes the archaeology *and* the restatement,
and sidesteps the dangling `Either` that trimming only the second clause would leave.

Also drop the trailing **"to rewrite it"**: nothing pins it, and "rewrite" presupposes an archive
already present in a superseded format — the same historical premise this issue exists to delete.

Result: `"MLflow run {run_id} has no {_MLFLOW_MODEL_ARTIFACT} artifact — re-materialise
\`trained_cv_model\` for this fold."`

## What changes, file by file

### `packages/ml_core/src/ml_core/base_forecaster.py`

Untouched by the merge — both hunks are exactly where the earlier reviews checked them.

- `_download_and_unpack_model`, `Raises:` entry (lines 83-86). A rewrite, not a substring deletion:

  ```text
  Raises:
      MlflowException: The run holds no model archive — re-materialise ``trained_cv_model``
          for this fold.
  ```

- The `MlflowException` message (lines 95-99): delete the middle sentence and "to rewrite it", per
  the draft above. The f-string's first line and the `from error` are untouched.

No change to control flow or exception type.

### `packages/ml_core/tests/test_base_forecaster.py`

`test_loading_a_run_with_no_archive_says_what_to_do_about_it` docstring (line 255): **delete** the
sentence "The case that matters is a run written before the model became a single archive
artifact:", and do not replace it. Line 255 ends in a colon and the sentence occupies it alone, so
the deletion leaves a grammatical docstring: the summary line already carries the why ("fails with
an actionable message, not MLflow's raw one"), and the following sentence about MLflow's raw error
being unactionable survives.

Explaining the empty-run mechanism here instead would copy an argument that lives in `cv_assets.py`
into a test docstring in a *different package*, where nothing can detect drift — squarely the "one
home per argument" rule in `code-style.md`. It would also be easy to get subtly wrong: `cv_metrics`
resolves a fold run but never calls `load_from_mlflow`, so it can leave an empty run behind but
cannot raise this error.

## Design-philosophy check

A prose-only diff moves no line: no control flow, no exception type, no asset check, no degradation
path. `_download_and_unpack_model` is reached from `load_from_mlflow` (R&D, which fails fast by
design) and from `fetch_model_artifacts` (promotion, which `inherent-stability.md` treats as a
promotion bug rather than a data outage). No principle is traded away and no engineering hypothesis
is in play.

## Tests

**No new test, and no assertion change.** The change is behaviour-preserving by construction, so
**nothing in this diff fails on `main` today** — correct for a prose-only change rather than a gap
to paper over. A test pinning the new wording would be a string literal copied from its own subject.

Four `pytest.raises(..., match=...)` sites touch the two messages, repo-wide. The merge added one of
them, so this inventory is fresher than the earlier reviews':

| Site | Assertion | Effect of this change |
|---|---|---|
| `test_base_forecaster.py:263` | ``match="re-materialise `trained_cv_model`"`` | must survive — see below |
| `test_base_forecaster.py:366` | ``match="re-materialise `trained_cv_model`"`` | **new in this merge**; same substring |
| `test_production_helpers.py:169` | `match="model_class"` | untouched (that message is now `main`'s) |
| `test_production_helpers.py:161` | `match="Materialise"` | untouched |

The first two are the ones that could break by accident: their `match` string sits inside the
message being edited. Backticks are not regex metacharacters and `pytest.raises` uses `re.search`,
so "re-materialise \`trained_cv_model\`" matches literally — and both deletions (the middle
sentence, and "to rewrite it") fall outside the matched span.

## Docs to update

**None.** No page or docstring quotes either message verbatim.

This issue completes no roadmap item, so there is no ship-time triage.

## The sweep

Re-run over the merged repo. **The merge introduced no new archaeology** and removed one of the
four sites. Three remain, all in scope and all fixed here: `base_forecaster.py:85` and `:97-98`,
and `tests/test_base_forecaster.py:255`.

One near-miss, judged **not** archaeology: `_production_helpers.py:139` says "renaming or removing
any of them in code leaves every model saved before the change unservable". That is a conditional
about any future rename — a present-tense statement of the mechanism, not a claim about a past state
of this repo.

**Outside `packages/ml_core/` — listed for Jack, not edited.** All three are comments in *tests*,
and all fall on the keep side of this rule: a runtime error path telling its reader to migrate
describes a world that does not exist, whereas a regression test's comment naming the bug it
prevents explains why the test is worth keeping — information not derivable from its assertions.

- `packages/contracts/tests/test_project_root.py:3-7` — "PROJECT_ROOT used to be
  `Path(__file__).parents[4]` … (issue #287)". Without it, a future reader may "simplify" it back.
- `tests/test_trained_cv_model.py` — "the exact input change that used to be rejected".
- `tests/test_assets.py` — "that it no longer rejects such a slice is pinned by …" and "What fails
  on `main` is the count", this repo's own test-writing convention rather than archaeology.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check && uv run pytest
```

Fast feedback on the affected module:

```bash
uv run pytest packages/ml_core/tests/test_base_forecaster.py
```

Because the whole diff is inside docstrings, the markdown lint that matters is the **docstring**
one — `.pre-commit-config.yaml:74-78` runs `scripts/lint_docstring_markdown.py` as a hook separate
from the `docs/` one, and CLAUDE.md's `pymarkdown scan` command covers only `docs/`, READMEs and
CLAUDE.md, none of which this branch touches:

```bash
uv run python scripts/lint_docstring_markdown.py packages/ml_core/src/ml_core/base_forecaster.py packages/ml_core/tests/test_base_forecaster.py
```

No network-gated tests are relevant, and no links change.

## Risks and open questions

Two of the four original open questions were **resolved by `main`**, both as recommended: the
`model_class` guard was kept (Q1), and promotion now applies the same servability check as serving
(Q3). The two below survive.

1. **`_download_and_unpack_model`'s remedy is wrong for one of its two callers** — and the merge
   sharpened this rather than fixing it. It is reached from `load_from_mlflow` (CV folds) and from
   `fetch_model_artifacts` (`_production_helpers.py:273`), which `promoted_model` calls with an
   operator-typed `mlflow_run_id`. An operator who mistypes a run id is told to "re-materialise
   `trained_cv_model` for this fold". `main` has now *pinned* that path with
   `test_base_forecaster.py:350-369`, whose own docstring calls it "the likeliest operator slip of
   the three" — so the mismatch is now tested-in rather than incidental. Fixing it means dropping
   the remedy (making the message true for both callers; `cv_power_forecasts` prints the same
   remedy itself, so the CV reader is not left empty-handed) or branching by caller. Either now
   costs **two** test `match` edits, not one. *Recommendation: not in this issue — it is a
   different defect from archaeology. Worth its own issue; I can file it.*

2. **Stamping `model_class` in the base class.** Make `BaseForecaster.save` concrete, delegating to
   an abstract `_save`, then stamp `class_target(self)` into `meta.json` itself, so the guard
   becomes unreachable. It buys deleting a small guard and its test; it costs `base_forecaster.py`,
   `xgboost_forecaster/forecaster.py`, three test classes and two docs pages. It gives up more than
   it did before this merge: `main` now reaches `CONFIG_CLASS` *through* `model_class`, so the
   field is more load-bearing, and forcing every future subclass to emit a `meta.json` for the base
   class to patch trades against design principle 5, "everything around the model is
   general-purpose" — while only narrowing "write `meta.json` with this field" to "write
   `meta.json`". *Recommendation: no, and not as its own issue either.*

## What the reviews changed

Four adversarial reviews ran before this merge — simplicity, correctness, then both again after the
previous merge under the updated brief that lets a reviewer propose a different architecture.

**Review 1 (simplicity)** cut a speculative clause from a proposed message, dropped a proposed test
assertion, trimmed the sweep, and caught that the plan's markdown-lint command scans `docs/` rather
than docstrings.

**Review 2 (correctness)** found the fold-run causal story backwards (`trained_cv_model` creates the
run *after* training) and a remedy decision wrong.

**Review 3 (simplicity, post-merge)** turned two of the plan's own fixes into deletions: the
`base_forecaster` message loses a whole redundant sentence rather than a repaired clause, and the
test docstring loses its archaeology with no replacement — the replacement would have copied an
argument across a package boundary and was itself imprecise about `cv_metrics`. It raised the open
questions above, two of which `main` has since implemented.

**Review 4 (correctness, post-merge)** confirmed the surviving reasoning — the redundant-sentence
deletion, the test-match survival, the hermeticity constraint traced end to end — and found three
precision defects, all since fixed.

**No fifth review for this merge.** The surviving scope is a strict subset of what reviews 3 and 4
already checked, and `base_forecaster.py:63-106` is byte-identical to the version they reviewed, so
their conclusions transfer exactly. The one genuinely new fact — the second `match` site at
`test_base_forecaster.py:366` — was checked directly and does not change the edit.

### Findings rejected

- **"Delete the `model_class` guard."** *Rejected in both simplicity reviews, and `main` has since
  agreed:* the guard survives inside `_check_meta_is_servable`.
- **"Cut the sweep section."** *Rejected as stated, applied in part.* The evidence table is gone and
  the out-of-scope hits are a list, but the sweep stays: the issue explicitly asks for it.

Every other finding across the four reviews was accepted.
