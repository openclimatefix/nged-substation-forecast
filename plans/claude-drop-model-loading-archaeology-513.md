# Plan: Drop backwards-compatibility archaeology from the model-loading error paths (#513)

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/513>
Branch: `claude/drop-model-loading-archaeology-513`

## Verdict

**Worth implementing, roughly as described.** Both docstrings and both error messages describe a
migration away from a state that exists nowhere: this project is greenfield, has no external
users, and holds no trained model predating either contract. That is squarely the
"[write about the present, not the past](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/)"
rule applied to code comments.

The change is small, entirely inside `packages/ml_core/`, and touches no behaviour. It is worth
doing now rather than later because both messages are read by an operator during an incident, and
a sentence about a code version that never existed is the worst possible thing to read at that
moment.

### Departures from the issue body

1. **Both guards stay.** The issue offers deleting the `model_class`-missing branch "and letting
   the `KeyError` speak" as a live alternative. Rejected — reasoning in the next section. This is
   a rewrite of four pieces of prose, not a deletion of any code.
2. **The "Out of scope" note is stale.** It says this is not part of
   [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228), which "only
   swaps the class-resolution mechanism inside these functions". #228 is **closed** — the swap has
   landed, and `load_forecaster_from_dir` already resolves via
   `contracts.config_schemas.import_class`. Nothing about the plan changes; the caveat simply no
   longer describes an open risk.
3. **One test assertion is strengthened**, which the issue does not ask for. See Tests.

## The per-branch decision

### `_production_helpers.load_forecaster_from_dir` — keep the guard, strip the archaeology

Keep it, for two reasons.

**A missing `model_class` is a present-tense contract violation, not a historical one.**
`BaseForecaster.save` is an `@abstractmethod`, and its docstring at
[`base_forecaster.py:224-230`](../packages/ml_core/src/ml_core/base_forecaster.py) makes stamping
`model_class` into `meta.json` a written requirement on **every** implementation. Today
`XGBoostForecaster` is the only one; the roadmap adds baseline forecasters and, later, a model
with a different `FeatureEngineer`. The subclass that forgets to honour the contract is the reader
this guard is for, and that reader is in the future, not the past.

**The `KeyError` alternative does not work as written.** The code reads
`meta.get("model_class")`. Deleting the branch therefore passes `None` into `import_class`, whose
first statement is `target.rpartition(".")` — so the operator gets
`AttributeError: 'NoneType' object has no attribute 'rpartition'` raised from inside
`contracts/config_schemas.py`, naming neither the file nor the field. Getting an actual `KeyError`
would mean *also* changing `.get` to `meta["model_class"]`, and even then the message is
`KeyError: 'model_class'` with no path and no remedy. This raise fires inside the `live_forecasts`
Dagster asset, where the reader is an operator reading a run log, not a developer at a REPL. Four
lines buying the path and the remedy is a good trade.

Rewrite the `Raises:` entry and the message so that they state the contract that was violated and
what to do, with **no claim about which code version wrote the file**. The message must name
`meta_path`, name `BaseForecaster.save` as the contract's owner, and give the remedy. Draft:

> `{meta_path}` has no 'model_class' field, so the concrete forecaster class cannot be
> reconstructed — `BaseForecaster.save` requires every implementation to stamp it. Re-materialise
> `promoted_model`; if that does not fix it, the forecaster's `save` is not honouring the
> contract.

Note the second clause: the issue's suggested wording ("re-materialise `promoted_model`") is right
only when the directory is stale or hand-assembled. If the *saving* class is at fault,
re-materialising re-downloads the same broken `meta.json`, so the message must not send the
operator round that loop indefinitely.

### `base_forecaster._download_and_unpack_model` — keep the guard, strip the archaeology

Keep it. The re-raise earns its place on a cause that has nothing to do with history:
`get_or_create_fold_run` ([`_mlflow_runs.py:108`](../packages/ml_core/src/ml_core/_mlflow_runs.py))
creates the fold's run **before** training, and `save_to_mlflow` uploads the archive at the end —
so a fold whose training crashed, was interrupted, or was cancelled leaves behind exactly this
state: a run that exists and holds no `model.tar.gz`. That is a routine R&D occurrence.

The value of the re-raise is recorded by the existing test's own name: MLflow's raw exception says
only that the artifact path was not found, which tells an operator nothing about which asset to
re-materialise.

Strip the clause "or it was saved before the model became a single archive artifact, in which case
the fold must be re-trained" from both the `Raises:` docstring entry and the message. What survives
is the live cause and the remedy: no model was ever saved to this run, so re-materialise
`trained_cv_model` for this fold. The remedy phrase is load-bearing — the existing test matches on
it — so it stays verbatim.

## What changes, file by file

### `packages/ml_core/src/ml_core/_production_helpers.py`

- `load_forecaster_from_dir`, `Raises:` entry for `ValueError` (lines 149-150): replace "it was
  saved by a code version predating this contract; re-promote with a version that stamps
  `model_class`" with a statement of the contract and the remedy.
- `load_forecaster_from_dir`, the `ValueError` message (lines 161-165): replace "Re-promote the
  model with a code version that stamps model_class (see BaseForecaster.save)" per the draft
  above. `BaseForecaster.save` stays cited — it is the contract, not a version.

No change to the control flow, the exception type, or the `.get` call.

### `packages/ml_core/src/ml_core/base_forecaster.py`

- `_download_and_unpack_model`, `Raises:` entry (lines 84-86): drop "or it was written before the
  model became a single archive artifact, in which case the fold must be re-trained"; keep "the run
  holds no model archive — nothing was ever saved to it".
- `_download_and_unpack_model`, the `MlflowException` message (lines 96-98): drop "or it was saved
  before the model became a single archive artifact"; keep "re-materialise `trained_cv_model` for
  this fold to rewrite it" **unchanged**.

No change to control flow or exception type.

### `packages/ml_core/tests/test_base_forecaster.py`

- `test_loading_a_run_with_no_archive_says_what_to_do_about_it` docstring (lines 206-212): the
  sentence "The case that matters is a run written before the model became a single archive
  artifact" is the same archaeology, in the place that explains *why the test exists*, so leaving
  it would defeat the point of the issue. Rewrite it to name the live cause — a fold run created
  before training that never received an archive, because training crashed or was interrupted.
  The parenthetical about `saved_run` being depended on for its tracking URI stays; it is a real,
  present-tense note about the fixture.

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
apply.

**Design principles:** nothing traded away. **Engineering hypotheses:** none cited — this change
delivers no falsifiable claim, it removes misleading prose.

## Tests

**There is deliberately no new test.** The change is behaviour-preserving by construction: same
control flow, same exception types, same trigger conditions. Inventing a test that pins the exact
new wording would fail the repo's own bar (a test whose only content is a string literal copied
from the source it tests asserts nothing about behaviour) and would make every future wording
improvement a two-file edit.

**One existing assertion is strengthened**, and it is the only assertion in this change that fails
on `main` today:

- `packages/ml_core/tests/test_production_helpers.py::test_load_forecaster_from_dir_raises_on_missing_model_class`
  currently asserts `match="model_class"` — satisfied by any message that merely names the field.
  Add an assertion that the message also names the asset to re-materialise (`promoted_model`).
  **Why it fails on `main`:** the current message says "Re-promote the model with a code version
  that stamps model_class" — it contains no `promoted_model`, so the assertion fails before the
  change and passes after.
  This pins a *property* — the message tells the operator which asset to act on — rather than the
  prose, so it survives future rewording that keeps the property.

The two existing tests that already pass in both states
(`test_load_forecaster_from_dir_raises_on_missing_dir`,
`test_loading_a_run_with_no_archive_says_what_to_do_about_it`) are the regression net proving the
guards still fire with the same exception types.

## Docs to update

**None.**
[`docs/live_service/operations.md:109-111`](../docs/live_service/operations.md) describes this
load path — "the concrete forecaster class is reconstructed from `meta.json`'s `model_class`
field" — and carries no archaeology, so it is already correct. It cites no error text verbatim, so
the message rewrite does not invalidate it.

This issue completes no roadmap item, so there is no ship-time triage: no "Implementation details"
section to delete, no status banner to move.

## The sweep

Swept the whole repo (`*.py`, `*.md`, `*.yaml`) for the pattern — error messages, comments and
docstrings describing migration away from a state the repo no longer has. Most matches for words
like "backwards", "historical" and "predates" are unrelated domain language (a backward-looking
delivery table, an ERA5 archive predating the ENS archive, a forecast run whose first 24 h predate
the chart window).

**Genuine hits inside this issue's scope — all three fixed by this plan:**

| Location | Text |
|---|---|
| `packages/ml_core/src/ml_core/_production_helpers.py:150` | "saved by a code version predating this contract" |
| `packages/ml_core/src/ml_core/base_forecaster.py:85, 97` | "written/saved before the model became a single archive artifact" |
| `packages/ml_core/tests/test_base_forecaster.py:208` | "a run written before the model became a single archive artifact" |

**Genuine hit outside `packages/ml_core/` — for Jack to rule on, not edited by this branch:**

- [`packages/contracts/tests/test_project_root.py:3-7`](../packages/contracts/tests/test_project_root.py)
  — the module docstring opens "PROJECT_ROOT used to be `Path(__file__).parents[4]` — a hard-coded
  directory depth that only held for an editable install, and silently resolved to the venv root
  under a non-editable install (issue #287)."
  **Recommendation: leave it.** A regression test's docstring naming the bug it prevents is a
  different genre from a runtime error path telling an operator to migrate. The value of these
  tests is not derivable from their assertions — without that sentence, a future reader cannot tell
  why the marker-based resolution is worth pinning, and may "simplify" it back to `parents[4]`.
  If Jack disagrees, it is a one-line reword in a package another session may own, so it belongs in
  its own change either way.

**Looked at and judged not archaeology:** `defs/production_assets.py:139-147` uses
`meta.get("model_class")` and friends for Dagster output metadata. That is defensive metadata
reporting on a directory the asset just wrote, with no claim about older formats. Noted only
because it shares the `.get`-instead-of-`[]` shape with the guard above.

## Verification commands

The green-before-push set:

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check && uv run pytest
```

Plus, because this change is entirely inside docstrings and messages:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

The two directly-affected test modules, for fast feedback:

```bash
uv run pytest packages/ml_core/tests/test_production_helpers.py packages/ml_core/tests/test_base_forecaster.py
```

No network-gated tests are relevant (nothing here touches NWP conversion), and no links change, so
`mkdocs build --strict` is not specifically needed beyond whatever CI runs.

## Risks and open questions

1. **Wave-3 overlap with [#512](https://github.com/openclimatefix/nged-substation-forecast/issues/512).**
   PR #532 edits `base_forecaster.py` at the import block (~line 14) and inside
   `BaseForecasterConfig` (~line 136). This plan edits lines 84-98. **No overlap** — a rebase, if
   needed, is textual and trivial. *Recommendation: implement without waiting; rebase on `main` if
   #532 lands first.*
2. **Is the guard-keeping call right?** Both decisions above land on "keep, reword", which is the
   less aggressive of the two options the issue offers. If Jack wants the `model_class` branch gone
   on the grounds that a subclass violating `BaseForecaster.save`'s contract should be caught by a
   test rather than at load time, that is a coherent position — but it wants a test asserting the
   contract across all `BaseForecaster` subclasses to replace it, which is a bigger change than
   this issue. *Recommendation: keep both guards as planned.*
3. **The strengthened assertion couples a test to the word `promoted_model`.** That is a Dagster
   asset name, not prose, so it changes only if the asset is renamed — at which point the test
   failing is correct. *Recommendation: include it; it is the only thing here that fails on `main`.*
