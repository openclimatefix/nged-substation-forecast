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

**The whole change is a net deletion.** Every replacement string is shorter than the one it
replaces; no line is added anywhere.

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

**Who actually reads this message.** A `meta.json` that exists but has no `model_class` has exactly
one reachable cause: a `BaseForecaster` subclass whose `save` does not honour the contract written
at `base_forecaster.py:224-230`. The directory is unpacked by `fetch_model_artifacts` from an
archive built by `save_to_mlflow` from a subclass's own `save`, so nothing else can produce that
shape. The *operator*-facing case — the asset was never materialised — is already covered by the
`FileNotFoundError` immediately above. So the reader is a **developer writing the next
`BaseForecaster` subclass**, not an operator during an incident.

**Keep it, on that reader's terms.** What that developer needs is which contract they broke and
where it is written. `KeyError: 'model_class'` raised at `_production_helpers.py:159` gives them a
field name and the *reader's* line number, and nothing about `BaseForecaster.save`; they have to go
read the function to work out what they failed to do. A one-sentence `ValueError` naming the path
and citing `BaseForecaster.save` gives it to them directly, for four lines. The guard is also
symmetric with the `FileNotFoundError` directly above it — same shape, same job — and deleting one
of a matched pair costs more in readability than it saves in lines.

Rewrite the `Raises:` entry and the message so they state the contract that was violated, with
**no claim about which code version wrote the file and no remedy**:

> `{meta_path}` has no 'model_class' field, so the concrete forecaster class cannot be
> reconstructed — `BaseForecaster.save` requires every implementation to stamp it.

**No remedy line, deliberately.** The obvious one ("re-materialise `promoted_model`") is wrong for
the only cause that can fire: re-materialising re-downloads the same `meta.json`, because the fault
is in the saving code. A message that offers an action which cannot work is worse than one that
just names the broken contract. (This is why the guard is *not* symmetric with the neighbouring
raise at `defs/production_assets.py:239-242`, which does carry a remedy — "Re-promote
`promoted_model` with a model that has trained boosters" is correct for *its* cause.)

Net effect: the new message is shorter than the one on `main`.

### `base_forecaster._download_and_unpack_model` — keep the guard, strip the archaeology

Keep it. The re-raise earns its place on a cause that has nothing to do with history:
`get_or_create_fold_run` (`_mlflow_runs.py:108`) creates the fold's run **before** training, and
`save_to_mlflow` uploads the archive at the end — so a fold whose training crashed, was
interrupted, or was cancelled leaves exactly this state: a run that exists and holds no
`model.tar.gz`. That is a routine R&D occurrence.

The value of the re-raise is recorded by the existing test's own name: MLflow's raw exception says
only that the artifact path was not found, which tells the reader nothing about which asset to
re-materialise.

Strip "or it was saved before the model became a single archive artifact, in which case the fold
must be re-trained" from both the `Raises:` entry and the message. The remedy phrase
("re-materialise `trained_cv_model` for this fold to rewrite it") is load-bearing — the existing
test matches on it — so it survives verbatim.

## What changes, file by file

### `packages/ml_core/src/ml_core/_production_helpers.py`

- `load_forecaster_from_dir`, `Raises:` entry for `ValueError` (lines 149-150): replace "it was
  saved by a code version predating this contract; re-promote with a version that stamps
  `model_class`" with a statement of the contract `BaseForecaster.save` imposes.
- The `ValueError` message (lines 161-165): replace "Re-promote the model with a code version that
  stamps model_class (see BaseForecaster.save)" per the draft above — `BaseForecaster.save` stays
  cited, as the contract rather than as a version.

No change to control flow, exception type, or the `meta.get(...)` call.

### `packages/ml_core/src/ml_core/base_forecaster.py`

- `_download_and_unpack_model`, `Raises:` entry (lines 84-86): drop "or it was written before the
  model became a single archive artifact, in which case the fold must be re-trained".
- The `MlflowException` message (lines 96-98): drop "or it was saved before the model became a
  single archive artifact".

**Mechanical trap:** the message opens `"Either no model was ever saved to this run, or ..."`.
Deleting the `or` arm leaves a dangling `Either`, so the opening must become `"No model was ever
saved to this run"` and the comma an em dash. The same applies to the `Raises:` entry, which reads
"either nothing was ever saved to it, or ...". Two words beyond the stated deletion, in each place.

No change to control flow or exception type.

### `packages/ml_core/tests/test_base_forecaster.py`

`test_loading_a_run_with_no_archive_says_what_to_do_about_it` docstring (lines 206-212): the
sentence "The case that matters is a run written before the model became a single archive artifact"
is the same archaeology, sitting in the place that explains *why the test exists* — leaving it
would defeat the point of the issue. Replace that clause (not the paragraph) with the live cause: a
fold run created before training that never received an archive. The rest of the docstring —
MLflow's raw error being unactionable, and the parenthetical about `saved_run` being depended on for
its tracking URI — is accurate present-tense text and stays.

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

The three existing tests that pass in both states are the regression net proving the guards still
fire with the same exception types and the same trigger conditions:

- `test_load_forecaster_from_dir_raises_on_missing_model_class` (`ValueError`, `match="model_class"`)
- `test_load_forecaster_from_dir_raises_on_missing_dir` (`FileNotFoundError`, `match="Materialise"`)
- `test_loading_a_run_with_no_archive_says_what_to_do_about_it`
  (`MlflowException`, `match="re-materialise \`trained_cv_model\`"`)

The third is the one that could break by accident: its `match` string sits inside the message being
edited, so it is the check that the archive guard's remedy phrase survived the clause deletion
intact.

## Docs to update

**None.** `docs/live_service/operations.md:109-111` describes this load path — "the concrete
forecaster class is reconstructed from `meta.json`'s `model_class` field" — carries no archaeology,
and cites no error text verbatim, so the rewrite does not invalidate it.

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
   whether four lines naming the violated contract are worth it for a developer writing the next
   `BaseForecaster` subclass. *Recommendation: keep, per the reasoning above.* If Jack prefers the
   deletion, the honest replacement is a test asserting every `BaseForecaster` subclass stamps
   `model_class` — which is a larger change than this issue, and should be its own.

## Review 1 — simplicity: findings rejected

Everything else the reviewer raised was accepted and is folded into the plan above (the message's
second clause cut, the test-assertion change dropped entirely, the dangling `Either`, the docstring
trimmed to a clause, the sweep compressed and its two misses added, the wrong lint command fixed).

- **"Delete the `model_class` guard; the plan is keeping code that only earns its keep as a
  migration aid."** *Rejected on the bottom line, accepted on the reasoning.* The reviewer was right
  that the plan's original justification (an operator reading a Dagster run log) was wrong, and
  right that the appended "if that does not fix it…" clause was evidence the remedy was mis-aimed;
  both are fixed above. But the guard is not only a migration aid — it also serves the developer
  writing the next `BaseForecaster` subclass, and `KeyError: 'model_class'` does not name the
  contract that was broken. See risk 2, which puts the call in front of Jack rather than burying
  it.
- **"Cut the sweep's evidence table."** *Rejected as stated, partly applied.* The per-file table is
  gone, but the sweep section stays: the issue explicitly asks for the sweep, and its results —
  one in-scope hit the issue did not name, three out-of-scope ones for Jack — are an output of this
  work, not plan bloat.
