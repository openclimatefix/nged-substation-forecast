# Plan: Drop backwards-compatibility archaeology from the model-loading error paths (#513)

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/513>
Branch: `claude/drop-model-loading-archaeology-513`

## Verdict

**Worth implementing, roughly as described.** Both docstrings and both error messages describe a
migration away from a state that exists nowhere: this project is greenfield, has no external
users, and holds no trained model predating either contract. That is squarely the
"[write about the present, not the past](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/)"
rule applied to code comments, and `code-style.md` sanctions the removal directly: comments may be
removed when they are "misleading or out of date".

The change is small, entirely inside `packages/ml_core/`, and touches no behaviour. It is worth
doing now rather than later because both messages are read while something is broken, and a
sentence about a code version that never existed is the worst thing to read at that moment.

**No line is added anywhere.** The `base_forecaster.py` edits are pure deletions; the
`_production_helpers.py` message swaps one sentence for a shorter one of the same shape.

### Departures from the issue body

1. **Both guards stay.** The issue offers deleting the `model_class`-missing branch "and letting
   the `KeyError` speak" as a live alternative. Rejected — reasoning in the next section.
2. **The "Out of scope" note is stale.** It says this is not part of
   [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228), which "only
   swaps the class-resolution mechanism inside these functions". #228 is **closed** — the swap has
   landed, and `load_forecaster_from_dir` already resolves via
   `contracts.config_schemas.import_class`. Nothing about the plan changes; the caveat simply no
   longer describes an open risk.

## The per-branch decision

### `_production_helpers.load_forecaster_from_dir` — keep the guard, strip the archaeology

**What actually causes it.** In production, a `meta.json` that exists but has no `model_class` has
one cause: a `BaseForecaster` subclass whose `save` does not honour the contract written at
`base_forecaster.py:224-230`. The directory is unpacked by `fetch_model_artifacts` from an archive
built by `save_to_mlflow` from a subclass's own `save`; a torn `shutil.move` leaves the directory
absent or the JSON truncated, which lands in the `FileNotFoundError` branch or a
`JSONDecodeError`, never in a well-formed `meta.json` missing one field. (A test fixture writes the
shape by hand — `test_production_helpers.py:163-166` — which is the point of that test.)

**Two readers, not one.** A developer writing the next `BaseForecaster` subclass will hit this at
development time, and needs to know which contract they broke. But an operator reads it too:
`inherent-stability.md:146` lists "the promoted model is empty or unloadable" as a hard failure
with "Human alerted? Yes, next business day", and `operations.md:279-283` carries a runbook section
headed "When the model fails to load", whose remedy is to re-promote or roll back.

**Keep the guard, and keep a remedy — a correct one.** `KeyError: 'model_class'` gives the
developer a field name and the *reader's* line number, nothing about `BaseForecaster.save`. The
guard is also symmetric with the `FileNotFoundError` four lines above — same shape, same job,
including its remedy — and breaking that symmetry costs more in readability than it saves in lines.
Rewrite the `Raises:` entry and the message to state the contract violated and the operator's
action, with **no claim about which code version wrote the file**:

> `{meta_path}` has no 'model_class' field, so the concrete forecaster class cannot be
> reconstructed — `BaseForecaster.save` requires every implementation to stamp it. Re-promote
> `promoted_model` with a different run, or roll back to the previous champion.

The remedy is "a *different* run", not "re-materialise": `PromotedModelConfig.mlflow_run_id`
(`production_assets.py:111-116`) takes an arbitrary run id, so promoting the previous champion —
the documented rollback — restores production immediately even when the fault is in a new
subclass's `save`. Re-promoting the *same* run id would re-download the same broken `meta.json`,
which is why the message must not say simply "re-promote". Wording tracks the runbook at
`operations.md:279-283` so an operator finds the same two options in both places.

### `base_forecaster._download_and_unpack_model` — keep the guard, strip the archaeology

Keep it. The re-raise earns its place on a cause that has nothing to do with history, but the
cause is **not** an interrupted training run: `trained_cv_model` creates the fold run *after*
training succeeds (`cv_assets.py:409` trains, `:422` `get_or_create_fold_run`, `:424`
`save_to_mlflow`), so a crashed fold leaves no run at all.

The live cause is that `get_or_create_fold_run` **creates** the run when no tagged one exists, and
two other assets call it themselves rather than waiting for `trained_cv_model`:
`cv_power_forecasts` resolves the fold run at `cv_assets.py:521` and calls
`load_from_mlflow(fold_run_id)` on the very next line, and `cv_metrics` does the same at
`cv_assets.py:916` under leaderboard scope. `deps=["trained_cv_model"]` is a lineage edge, not a
runtime interlock, so materialising `cv_power_forecasts` for a fold whose `trained_cv_model` has
not run — the normal single-asset materialisation from the Dagster UI — creates an empty fold run
and fails on it immediately. That makes the surviving remedy exactly right: re-materialise
`trained_cv_model` for this fold.

The value of the re-raise is recorded by the existing test's own name: MLflow's raw exception says
only that the artifact path was not found, which tells the reader nothing about which asset to
re-materialise.

Strip "or it was saved before the model became a single archive artifact, in which case the fold
must be re-trained" from both the `Raises:` entry and the message. Also drop the trailing "**to
rewrite it**": nothing pins it (the test matches only ``re-materialise `trained_cv_model` ``), and
"rewrite" presupposes an archive already present in a superseded format — the same historical
premise this issue exists to delete. The message ends "…re-materialise `trained_cv_model` for this
fold.", which is verbatim the remedy `cv_power_forecasts` already gives at `cv_assets.py:526-527`.

## What changes, file by file

### `packages/ml_core/src/ml_core/_production_helpers.py`

- `load_forecaster_from_dir`, `Raises:` entry for `ValueError` (lines 149-150): replace "it was
  saved by a code version predating this contract; re-promote with a version that stamps
  `model_class`" with the contract `BaseForecaster.save` imposes, plus the rollback remedy.
- The `ValueError` message (lines 161-165): replace "Re-promote the model with a code version that
  stamps model_class (see BaseForecaster.save)" per the draft above — `BaseForecaster.save` stays
  cited, as the contract rather than as a version.

No change to control flow, exception type, or the `meta.get(...)` call.

### `packages/ml_core/src/ml_core/base_forecaster.py`

- `_download_and_unpack_model`, `Raises:` entry (lines 84-86): drop "or it was written before the
  model became a single archive artifact, in which case the fold must be re-trained".
- The `MlflowException` message (lines 96-98): drop "or it was saved before the model became a
  single archive artifact", and drop the trailing "to rewrite it".

**Mechanical trap:** the message opens `"Either no model was ever saved to this run, or ..."`.
Deleting the `or` arm leaves a dangling `Either`, so the opening must become `"No model was ever
saved to this run"` and the comma disappears — the em dash before `re-materialise` is already
there. The same applies to the `Raises:` entry, which reads "either nothing was ever saved to it,
or ...".

No change to control flow or exception type.

### `packages/ml_core/tests/test_base_forecaster.py`

`test_loading_a_run_with_no_archive_says_what_to_do_about_it` docstring (lines 206-212): the
sentence "The case that matters is a run written before the model became a single archive artifact"
is the same archaeology, sitting in the place that explains *why the test exists* — leaving it
would defeat the point of the issue. Replace that clause (not the paragraph) with the live cause
established above: `cv_power_forecasts` and `cv_metrics` resolve the fold run by tag and **create**
it if absent, so materialising either for a fold whose `trained_cv_model` has not run yields a run
holding no archive. The rest of the docstring — MLflow's raw error being unactionable, and the
parenthetical about `saved_run` being depended on for its tracking URI — is accurate present-tense
text and stays.

Getting this right is the point of the exercise: the issue exists to stop misleading prose sitting
in these error paths, and replacing a false statement about the past with a false statement about
the present would be a worse outcome than leaving it alone.

The `pytest.raises(..., match="re-materialise \`trained_cv_model\`")` assertion is untouched.

## Design-philosophy check

**Which side of the fail-fast line is each path on?**

- `load_forecaster_from_dir` runs in **production** (`live_forecasts`, via
  `defs/production_assets.py:237`), where the standing rule is to degrade rather than raise. This
  raise is nevertheless correct and stays: CLAUDE.md's inherent-stability section names "an empty
  promoted model, a **contract violation**" as exactly the class of state production *should* raise
  on, because it is our own bug rather than the outside world misbehaving. A `meta.json` that does
  not honour `BaseForecaster.save`'s documented contract is our bug. This plan does not move the
  line either way — it changes prose only.
- `_download_and_unpack_model` is reached from `load_from_mlflow` (R&D, CV) and from
  `fetch_model_artifacts` (promotion). R&D fails fast by design.

**Asset checks:** none added or edited, so the `WARN`/`blocking=False`/cannot-raise rules do not
apply. **Design principles:** nothing traded away. **Engineering hypotheses:** none cited — this
change delivers no falsifiable claim, it removes misleading prose.

## Tests

**No test assertion changes, and no new test.** The change is behaviour-preserving by construction:
same control flow, same exception types, same trigger conditions. The honest consequence is that
**nothing in this diff fails on `main` today** — and that is correct for a prose-only change rather
than a gap to paper over. Inventing an assertion that pins the new wording would fail the repo's own
bar: a test whose content is a string literal copied from the source it tests asserts nothing about
behaviour, and makes every future wording improvement a two-file edit.

The three existing tests that pass in both states are the regression net, and they are stronger
than "nothing fails" suggests. `KeyError` is not a subclass of `ValueError`, so
`test_load_forecaster_from_dir_raises_on_missing_model_class` already fails if the guard is deleted
— it is a genuine behavioural pin on the guard's *existence*, not merely on its wording. A repo-wide
grep confirms these are the only three `pytest.raises(..., match=...)` sites touching either
message; nothing in root `tests/` matches on them.

- `test_load_forecaster_from_dir_raises_on_missing_model_class` (`ValueError`, `match="model_class"`)
- `test_load_forecaster_from_dir_raises_on_missing_dir` (`FileNotFoundError`, `match="Materialise"`)
- `test_loading_a_run_with_no_archive_says_what_to_do_about_it`
  (`MlflowException`, `match="re-materialise \`trained_cv_model\`"`)

The third is the one that could break by accident: its `match` string sits inside the message being
edited, so it is the check that the archive guard's remedy phrase survived the clause deletion
intact. Backticks are not regex metacharacters, so it matches literally — the trailing "to rewrite
it" that this plan also deletes sits outside the matched substring and is free to go.

## Docs to update

**None.** No page or docstring quotes either message verbatim, so nothing goes stale. Checked:

- `docs/live_service/operations.md:109-111` describes this load path — "the concrete forecaster
  class is reconstructed from `meta.json`'s `model_class` field" — with no archaeology and no error
  text.
- `docs/live_service/operations.md:279-283` ("When the model fails to load") is the runbook the
  rewritten `ValueError` now echoes. It already says re-promote or roll back, so it needs no edit —
  but the implementer should re-read it to keep the message's two options in the same order.
- `docs/design-philosophy/inherent-stability.md:146`, `docs/architecture/ml-orchestration.md:57`,
  `docs/live_service/aws.md:642` and `docs/live_service/operations.md:62-63` all reference these
  code paths without citing error text.
- `defs/checks.py:739-775` reads the same `meta.json` and *degrades* rather than raising, which is
  correct for an asset check and independent of this change.

This issue completes no roadmap item, so there is no ship-time triage.

## The sweep

Swept the repo (`*.py`, `*.md`, `*.yaml`) for the pattern: error messages, comments and docstrings
describing migration away from a state the repo no longer has. Most matches for "backwards",
"historical" and "predates" are unrelated domain language (a backward-looking delivery table, an
ERA5 archive predating the ENS archive, a forecast run whose first 24 h predate the chart window).

**In scope, fixed by this plan:** the two source sites named in the issue
(`_production_helpers.py:150`, `base_forecaster.py:85` and `:97`) plus one the issue did not name,
`packages/ml_core/tests/test_base_forecaster.py:208`.

**Outside `packages/ml_core/` — listed for Jack, not edited by this branch.** All three are
comments in *tests*, and the rule separating them from the two source sites above is: a runtime
error path that tells its reader to migrate is describing a world that does not exist, whereas a
regression test's comment naming the bug it prevents explains why the test is worth keeping —
information that is not derivable from its assertions. All three fall on the keep side by that
rule; Jack may disagree, and each is a one-line reword in a package another session may own.

- `packages/contracts/tests/test_project_root.py:3-7` — "PROJECT_ROOT used to be
  `Path(__file__).parents[4]` … (issue #287)". *Recommendation: leave.* Without it, a future reader
  cannot tell why marker-based resolution is worth pinning and may "simplify" it back.
- `tests/test_trained_cv_model.py:272` — "the exact input change that used to be rejected".
  *Recommendation: leave.* It explains why that specific second-pass input was chosen.
- `tests/test_assets.py:365` — "that it no longer rejects such a slice is pinned by …" and "What
  fails on `main` is the count". *Recommendation: leave.* The "what fails on `main`" sentence is
  this repo's own test-writing convention, not archaeology.

## Verification commands

The green-before-push set:

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check && uv run pytest
```

The two directly-affected test modules, for fast feedback:

```bash
uv run pytest packages/ml_core/tests/test_production_helpers.py packages/ml_core/tests/test_base_forecaster.py
```

Because the whole diff is inside docstrings, the markdown lint that matters is the **docstring**
one, not the `docs/` one — `.pre-commit-config.yaml` runs them as two separate hooks, and the
`pymarkdown scan` command in CLAUDE.md covers only `docs/`, READMEs and CLAUDE.md, none of which
this branch touches:

```bash
uv run python scripts/lint_docstring_markdown.py packages/ml_core/src/ml_core/_production_helpers.py packages/ml_core/src/ml_core/base_forecaster.py packages/ml_core/tests/test_base_forecaster.py
```

No network-gated tests are relevant, and no links change.

## Risks and open questions

1. **Wave-3 overlap with [#512](https://github.com/openclimatefix/nged-substation-forecast/issues/512).**
   PR #532 edits `base_forecaster.py` at the import block (~line 14) and inside
   `BaseForecasterConfig` (~line 136). This plan edits lines 84-98. **No overlap** — a rebase, if
   needed, is textual and trivial. *Recommendation: implement without waiting; rebase on `main` if
   #532 lands first.*
2. **Is keeping the `model_class` guard right?** The alternative — `meta["model_class"]` and let
   the `KeyError` fly — is about 13 lines smaller and was pressed hard in review. It turns on
   whether four lines naming the violated contract and the rollback are worth it for the two
   readers above. *Recommendation: keep, per the reasoning above.* If Jack prefers the deletion,
   the honest replacement is a test asserting every `BaseForecaster` subclass stamps `model_class`
   — a larger change than this issue, and its own.

## What the two reviews changed

**Review 1 (simplicity)** cut the message's speculative second clause, dropped a proposed test
assertion entirely, trimmed the test-docstring rewrite to a clause, compressed the sweep, caught
the dangling `Either`, and caught that the plan's markdown-lint command scans `docs/` rather than
docstrings.

**Review 2 (correctness)** found the fold-run causal story backwards and the remedy decision wrong
— both are rewritten above. Its checks that came back clean are worth recording: every line number
and quoted string the plan attributes to source is correct; the three `pytest.raises` match sites
are the only ones repo-wide; no doc page quotes either message.

### Findings rejected

- **Review 1 — "delete the `model_class` guard; it only earns its keep as a migration aid."**
  *Rejected on the bottom line, accepted on the reasoning.* The reviewer was right that the plan's
  original justification was wrong and that an appended "if that does not fix it…" clause was
  evidence the remedy was mis-aimed. But the guard is not only a migration aid, and
  `KeyError: 'model_class'` names neither the contract nor the operator's action. Surfaced as
  risk 2 rather than buried.
- **Review 1 — "cut the sweep's evidence table."** *Rejected as stated, partly applied.* The
  per-file table is gone, but the sweep section stays: the issue explicitly asks for the sweep, and
  its results are an output of this work, not plan bloat.

Every other finding from both reviews was accepted.
