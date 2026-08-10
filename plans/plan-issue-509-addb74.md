# Plan — issue #509: `ecmwf_ens`'s in-asset checks run after the Delta write

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/509>
Branch: `claude/plan-issue-509-addb74`

## Verdict

**Worth implementing, but re-scoped and with one factual correction to the issue.** The rule-7
hazard is real and cheap to close. The issue's second bullet — "deciding what the write should do
about a partition that already exists is the more important half of this issue" — is *already
tracked in full elsewhere* and must not be settled here.

### Departures from the issue body

1. **The write-idempotency half goes to #476, not here.**
   [#476](https://github.com/openclimatefix/nged-substation-forecast/issues/476) ("Add
   `replace_partition` support to `write_nwp` so an incomplete NWP run can be corrected in place")
   already owns exactly that question, including the `Timestamp`-partition-predicate wrinkle and
   the "how does an operator trigger a replace without making the daily path overwritable"
   decision. [#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501) lists
   landing #476 as one of its options, and
   [`docs/live_service/operations.md`](../docs/live_service/operations.md) already carries the
   operator rule ("Do not re-materialise a partition that has already landed") pointing at #476.
   Doing half of #476 here would pre-empt a design decision #476 exists to make.

   **Action — and this is *not* a plain "subsumed by #476".** #476's stated scope is the *replace*
   path (how an operator overwrites a landed partition, and the `Timestamp`-predicate verification
   it needs). It does **not** cover the narrower question in Risks Q1 below: should `write_nwp`
   *refuse* to append onto a partition that already exists? Closing #509 while saying "the write
   half is #476's" would leave that question tracked nowhere. So: comment on **#476** adding the
   refusal guard to its "What to build" (it is the same decision as "day-to-day materialisations
   stay strictly append-only by default", which #476 already asserts), then comment on #509
   recording the split, and let #509 close on the rule-7 fix alone.

2. **The issue asks "should the checks move above the write?" as an either/or with the
   `try`/`except`. It is both, and neither alone is sufficient.**
   - *Reordering alone* stops the duplication but leaves a bug in the warning path failing the
     ingest outright — which is still exactly rule 7 ("never let the warning path be able to fail
     the thing it is warning about"), just with a cheaper consequence.
   - *A `try`/`except` alone* stops the run failing, but leaves every *other* post-write statement
     (the two `AssetCheckResult` builders, `_nwp_null_slices_metadata`, `_nwp_run_shape_metadata`)
     able to fail after the append has committed.

   So: assess **and** build the check results and metadata **before** `write_nwp`, all under the
   house catch-all. After that, the only thing between the append and the return is constructing a
   `MaterializeResult` from already-computed values.

3. **Clarification: the 8-retry ladder cannot fire after the write, so this is not an
   automatic-retry hazard.** The issue's title ("…duplicates NWP rows on retry") and its
   "The partition is then retried" read as if the retry ladder applies to a post-write raise — its
   own parenthetical already scopes the ladder correctly and names the manual route, so this is a
   sharpening of the issue rather than a correction of it, and a PR body should not claim
   otherwise. It does not: `RetryRequested` is
   raised only inside the `try` block wrapping `open_ecmwf_ens_run` / `download_ecmwf_ens_data` /
   `convert_nwp_xarray_dataset_to_polars_dataframe`
   ([`assets.py:271-283`](../src/nged_substation_forecast/defs/assets.py)), strictly *before*
   `write_nwp`. There is no `RetryPolicy` on the asset or on `ecmwf_ens_job`, and no `run_retries`
   block in any `dagster.yaml` documented for either deployment (searched `docs/live_service/`,
   `docs/architecture/production-deployment.md`). So a raise in the assessment fails the run, pages
   via `sentry_capture_failure`, and duplicates rows **only when a human re-materialises the red
   partition** — which is precisely what an operator would do, so the hazard is real, but it is
   operator-triggered rather than automatic. The plan (and any PR body) should say so rather than
   repeating the issue's framing.

4. **Reordering is free — nothing is lost by assessing the in-memory frame before the write.**
   Worth stating because it is the obvious objection: `live_forecasts_are_healthy` deliberately
   reads back *off disk* so it can catch "the run succeeded but wrote nothing usable". These two
   checks never did that — both read only the in-memory `nwp` frame (completeness also takes the
   scalar `expected_n_h3_cells`, derived from `h3_grid`, not from the table). And `write_nwp` cannot
   change what they would report, for a stronger reason than "rounding preserves nulls and sorting
   preserves counts": `write_nwp` never mutates its argument at all — it builds a `rounded` copy
   and hands an Arrow table to delta-rs. The pre-write and post-write reports are computed from the
   same object.

## What changes, file by file

### `src/nged_substation_forecast/defs/assets.py`

- **`ecmwf_ens`** — reorder the body to: download/convert (unchanged) → assess → `write_nwp` →
  `return MaterializeResult(...)` built from the already-computed check results and metadata. Add a
  comment at the write naming *why* the order matters: `write_nwp` is `mode="append"` with no
  dedup, so anything that can fail after the append turns a bug into duplicated primary keys on the
  operator's inevitable re-materialisation (cross-reference #476 for the write-side fix).
- **Two new guarded helpers**, one per declared check, so a bug in one does not blind the other:
  - `_assess_nwp_quality_or_degraded(nwp, context) -> AssetCheckResult`
  - `_assess_nwp_run_completeness_or_degraded(nwp, expected_n_h3_cells, context) -> tuple[AssetCheckResult, dict[str, MetadataValue]]`

  Each wraps its existing `assess_*` call *and* its existing `_nwp_*_check_result` builder (and,
  for completeness, `_nwp_run_shape_metadata`) in the catch-all already used twice in
  [`defs/checks.py`](../src/nged_substation_forecast/defs/checks.py): `except BaseException`,
  re-raise `KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError`, log the traceback,
  `report_check_degradation(<check name>, exc)`, return a `passed=False`,
  `AssetCheckSeverity.WARN` result whose description names the failure (include `{exc!r}`, matching
  `live_forecasts_are_healthy`). Do not re-derive the reasoning in a comment — point at the
  existing one in `power_data_is_fresh`, per the repo's doc-link rule.
- **Log through `context.log`, not a module logger.** `defs/assets.py` has no module-level `logger`
  (unlike `checks.py:86`), and the runbook tells operators the full traceback is "in the run's
  logs", which `context.log` guarantees. Confirm `DagsterLogManager.exception` exists on the pinned
  version; fall back to `context.log.error(..., exc_info=True)` if not. This is why the helpers take
  `context` rather than a bare logger.
- **Hard constraint on the degraded path — it must emit *both* check names, and each result must
  carry an explicit `check_name`.** Verified by execution against the pinned Dagster: an asset that
  declares two `check_specs` and returns a `MaterializeResult` carrying only one `AssetCheckResult`
  **fails the step** (`success=False`, no check evaluations recorded at all). Independent guards
  give the "both" half for free. The second half is a live trap: the fallback this plan points at,
  `power_data_is_fresh`'s at
  [`checks.py:333-337`](../src/nged_substation_forecast/defs/checks.py), passes **no**
  `check_name`, because a standalone `@asset_check` does not need one — and an `AssetCheckResult`
  with no `check_name` inside a multi-check asset fails the step identically
  (`DagsterInvariantViolationError` from `resolve_target_check_key`). So each fallback must set
  `check_name=_NWP_QUALITY_CHECK_NAME` / `check_name=_NWP_COMPLETENESS_CHECK_NAME` explicitly.
  Copying the model fallback verbatim would turn the degraded path into exactly the step failure
  this change exists to prevent.
- **Degraded materialisation metadata:** on the completeness path degrading, *omit* the five shape
  keys (`n_ensemble_members`, `n_valid_times`, `n_h3_cells`, `valid_time_min`, `valid_time_max`)
  rather than emitting sentinel values. `_nwp_run_shape_metadata`'s existing comment warns that a
  key whose metadata *type* changes between runs breaks the Dagster UI's timeline plot; a missing
  key is a gap in that plot, which is honest, whereas `-1` is a plotted lie. `n_rows`, `path` and
  `init_time` are computed outside the guard and are always published.
- Add the `DagsterExecutionInterruptedError` import (and `report_check_degradation` from
  `nged_substation_forecast._sentry`) to this module.

### `src/nged_substation_forecast/_sentry.py`

Two places make the same now-wrong claim, and both must change:

- `report_check_degradation`'s docstring (`:136-137`) says "Both of this function's callers — the
  standalone `@asset_check`s `power_data_is_fresh` and `live_forecasts_are_healthy`…". The new
  callers are *in-asset* check results, not standalone `@asset_check`s, and the "the run no longer
  fails and `sentry_capture_failure` no longer fires" argument holds for them identically
  (`ecmwf_ens_job` carries the same hook).
- The **module** docstring (`:9-11`) narrows the covered fault to "a standalone `@asset_check` that
  caught its own exception". Widen it to any guarded warning path.

Rewrite both to describe the current set, without narrating the change.

## Design-philosophy check

- **Production path, so degrade.** `ecmwf_ens` is a scheduled production ingest
  (`ecmwf_ens_schedule`, 08:30 UTC, `sentry_capture_failure` hook). This change is squarely
  [rule 7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules),
  and it strengthens
  [rule 6](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules):
  both checks stay `AssetCheckSeverity.WARN` / `blocking=False`, and after this change nothing
  either check's *body* does can fail its own step. State it with that carve-out rather than
  absolutely: what remains outside every guard is Dagster serialising the returned results into
  events, and process death — the same residual `checks.py`'s module docstring already names. Here
  that residual is worth one honest sentence rather than silence, because for a standalone check it
  costs a failed run whereas here it costs duplicated NWP rows on the operator's re-run.
- **No principle is traded away.** No new failure mode is introduced: a run that would have
  materialised still materialises, with the same results, on every non-degraded path.
- **Hypotheses.** This is an
  [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
  change, measured by **T1.1** (interventions per quarter, classified by cause — a warning-path bug
  that duplicates NWP rows is a manual-intervention event of the worst kind, because the recovery
  is a data repair, not a re-run) and touched by **T1.4** (operability by a non-expert, since the
  runbook line about re-materialising a failed partition changes).
- **Liberal about missing inputs, strict about malformed ones** is unaffected: `Nwp.validate` still
  runs in the converter, still before the write, and still rejects malformed runs.

## Tests

All in `tests/test_assets.py`, in the existing `--- ecmwf_ens ---` section, reusing `_make_nwp`,
`_write_h3_grid_weights` and `_check_evaluations`.

1. **`test_ecmwf_ens_lands_the_run_when_the_quality_assessment_raises`**
   `monkeypatch.setattr(assets, "assess_nwp_quality", <raises RuntimeError>)`, materialise.
   Asserts: `result.success` is True; `pl.read_delta(...).height == 4` (written exactly once);
   `_check_evaluations(result)["nwp_has_no_unexpected_nulls"]` exists, `passed is False`,
   `severity == AssetCheckSeverity.WARN`; and `["nwp_run_is_complete"]` is still *evaluated* (the
   two guards are independent).
   **Fails on `main`:** the assessment runs post-write today, so the `RuntimeError` propagates out
   of the asset. Note the precise mechanism, because the obvious phrasing is wrong: `materialize()`
   defaults to `raise_on_error=True` (every existing test here relies on that), so on `main` the
   exception escapes the `materialize(...)` call itself and the test **errors** — it never reaches
   `assert result.success`. Still a genuine main-failing test; do not describe it in a PR body as
   "`result.success` is False".

2. **`test_ecmwf_ens_lands_the_run_when_the_completeness_assessment_raises`**
   Same shape, patching `assets.assess_nwp_run_completeness`. Additionally asserts that
   `nwp_has_no_unexpected_nulls` still **passed** (a clean frame), proving the failure domains are
   separate, and that the materialisation metadata still carries `n_rows` while the five shape keys
   are absent.
   **Fails on `main`:** the raise propagates, by the same `raise_on_error=True` mechanism as above.

3. **`test_ecmwf_ens_assesses_before_writing`** — the test that actually pins *this issue's* fix
   rather than rule 7 in general. Patch `assets.assess_nwp_quality` to raise
   `DagsterExecutionInterruptedError` (the one class the guard deliberately re-raises, so it
   escapes as a cancelled run would). Call the asset directly via `build_asset_context(...)` +
   `pytest.raises`, mirroring `test_ecmwf_ens_retries_when_run_not_yet_available`, and assert
   `not Path(Settings().nwp_data_path).exists()` — nothing landed.
   **Fails on `main`:** today the assessment runs after `write_nwp`, so the Delta table exists with
   4 rows when the exception escapes.
   *Why this shape:* the ordering is not observable through an ordinary raise once the guard is in
   place, and the alternative (a call-order recorder around `write_nwp`) asserts an implementation
   detail rather than the invariant we care about. Using the re-raised class tests the ordering and
   documents the cancellation carve-out in one go. Note the `code-style`/`checks.py` warning that a
   `BaseException` guard also swallows `pytest.fail`/`pytest.skip` — never use one as a
   "must not be called" sentinel inside these bodies; assert after the call.

4. **Existing tests that must keep passing unchanged** —
   `test_ecmwf_ens_materialises_and_appends_nwp`,
   `test_ecmwf_ens_warns_on_scattered_nulls_but_still_materialises`,
   `test_ecmwf_ens_warns_on_incomplete_run_but_still_materialises`,
   `test_ecmwf_ens_retries_when_run_not_yet_available`, and the wholly-missing-variable retry test
   (whose `assert not Path(Settings().nwp_data_path).exists()` is the pre-existing statement of the
   same invariant one stage earlier). If any of these needs editing, that is a signal the change
   altered behaviour it should not have.

No new tests in `packages/delta_store/tests/test_nwp.py`: `write_nwp` is untouched here (that is
#476).

## Docs to update

Written to describe how the code works *now*, per CLAUDE.md's "write about the present".

- **[`docs/live_service/operations.md`](../docs/live_service/operations.md)** — in the
  "Do not re-materialise a partition that has already landed" block, add the counterpart the
  operator actually needs: a **failed** `ecmwf_ens` run wrote nothing, because validation *and*
  both quality assessments now run before the Delta append, so re-running a failed partition is
  safe. State the one residual caveat plainly: a process killed between the Delta commit and
  Dagster recording success leaves a red partition with rows on disk, so check the table before
  re-running a partition that failed for an infrastructure reason rather than a code one.
- **[`docs/live_service/operations.md`](../docs/live_service/operations.md), second edit** — the
  page teaches the operator to read `power_data_is_fresh`'s degraded description ("`Could not
  evaluate power-data freshness: …` is the check reporting that it could not read its own
  inputs … also sent to Sentry tagged `asset_check:power_data_is_fresh`"). Both NWP checks gain
  that same state, and the NWP sections say nothing about it. Add the equivalent line there —
  otherwise the change ships a new operator-visible check state with no runbook entry, which is
  exactly what **T1.4** measures.
- **[`docs/architecture/production-deployment.md`](../docs/architecture/production-deployment.md)**
  — this is the most concrete miss the review found. The page states outright: "(The two NWP checks
  are computed inside the `ecmwf_ens` asset and **have no catch-all**, so a raise there does fail
  the run and the hook does see it.)" That parenthetical becomes false, and it is the sentence
  explaining why `report_check_degradation` has exactly two callers. Rewrite it.
- **[`docs/architecture/ecmwf-ens-known-issues.md`](../docs/architecture/ecmwf-ens-known-issues.md)**
  — the sentence "A run that fails ingest writes nothing (validation runs before the Delta append),
  so there are no partial partitions to clean up". Do **not** rewrite it to "validation and both
  assessments run before the append": after this change an assessment can no longer fail ingest at
  all, so citing it as a reason a *failed* ingest wrote nothing muddles the sentence. Keep
  validation as that reason, and add a separate sentence that the assessments now precede the
  append too, so no warning-path bug can land a second copy of a run.
- **[`docs/design-philosophy/inherent-stability.md`](../docs/design-philosophy/inherent-stability.md)**
  — rule 7 currently names `power_data_is_fresh` and `live_forecasts_are_healthy` as the checks
  running under a catch-all. Add the two in-asset `ecmwf_ens` checks, and (this is the part worth
  writing down, because it is the generalisation this issue produced) that a warning path
  computed *inside* an asset must also run **before** the asset's non-idempotent write, not merely
  under a guard — otherwise a warning-path bug corrupts the data instead of merely failing the run.
- **[`src/nged_substation_forecast/defs/checks.py`](../src/nged_substation_forecast/defs/checks.py)**
  module docstring — its "Both checks are `AssetCheckSeverity.WARN`…" paragraph is scoped to that
  module's two standalone checks, so it is probably still accurate; the implementer should read it
  and only touch it if it reads as an inventory of every guarded warning path in the repo.
- **No roadmap ship-time triage.** #509 does not complete a roadmap item; nothing in `docs/roadmap/`
  references it, and no "Implementation details" section is retired by it.
- **`docs/architecture/why-dagster-not-airflow.md:202`** lists the check names in a table cell — no
  change needed (names are unchanged), noted only so the reviewer does not flag it as missed.

## Verification commands

The green-before-push set:

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check && uv run pytest
```

Plus, for this change specifically:

```bash
uv run pytest tests/test_assets.py -k ecmwf_ens -v
```

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

```bash
uv run mkdocs build --strict
```

`--run-network` is **not** needed: nothing here touches the convention-sensitive ECMWF conversion
path. `mkdocs build --strict` *is* needed because three docs pages change, and the rendered HTML for
the edited `operations.md` and `inherent-stability.md` sections should be read (nested list items —
see the `mkdocs-authoring` skill).

## Risks and open questions

1. **Should `write_nwp` also refuse to append to a partition that already exists?** It is a
   four-line guard and would close the residual hazard (the killed-process case, and the operator
   who re-runs a landed partition despite the runbook).
   **Recommendation: no, not here.** #476 has to decide how a replace is triggered, and a hard
   refusal added first would either be immediately relaxed by #476 or would block the legitimate
   replace path. Doing it in #509 also breaks the "stay inside the issue's scope" rule. If Jack
   wants it sooner, the right move is to prioritise #476, not to widen #509.

2. **One Sentry event per degraded check, or one per exception?** Two independent guards means two
   `report_check_degradation` calls if a shared cause takes out both.
   **Recommendation: two.** The `check_name` tag is documented as "the Dagster asset-check name…
   so events can be filtered per check", so inventing a combined name would break that contract;
   Sentry groups by exception anyway, so the duplication costs a tag, not an alert storm.

3. **Should the degraded check description carry `repr(exc)`?** `live_forecasts_are_healthy` does
   (`f"Could not evaluate live-forecast health: {exc!r}"`).
   **Recommendation: yes, match it** — an NWP frame's exceptions carry column names and counts, not
   credentials, and the Dagster Checks view is the first place an operator looks.

4. **Does this make the "a red `ecmwf_ens` partition means nothing landed" claim safe to state
   flatly in the runbook?** Almost — see the killed-process caveat above, which is why the runbook
   edit states the caveat rather than the flat claim. Flagging it because it is the kind of
   simplification a reviewer may push for.

## Review findings (step 5)

A fresh sub-agent reviewed this plan with no access to the reasoning that produced it. It confirmed
the four load-bearing claims about current behaviour (assessments run post-write; `write_nwp` is
append-only with no dedup; the retry ladder cannot fire post-write and nothing auto-retries; and
pre- vs post-write assessment is identical), and independently reproduced the two-`check_specs`
Dagster experiment. Each finding below was re-verified against the code before being applied.

### Accepted and folded in

1. **The fallback `AssetCheckResult` must set `check_name` explicitly.** The model fallback
   (`checks.py:333-337`) omits it because a standalone check does not need one; inside a
   multi-check asset an omitted `check_name` fails the step exactly like an omitted result.
   Verified: `checks.py:333-337` has no `check_name`. This was the one finding that would have
   shipped a fail-closed degraded path.
2. **`defs/assets.py` has no module-level `logger`.** Verified (`getLogger` appears only at
   `checks.py:86`). Resolved by logging through `context.log`, which also matches the runbook's
   "the traceback is in the run's logs".
3. **`docs/architecture/production-deployment.md` explicitly asserts the two NWP checks "have no
   catch-all".** Verified at `:174-176`. Was missing from the docs list; added.
4. **`_sentry.py`'s *module* docstring makes the same claim as the function docstring.** Verified
   at `:9-11`. Only the function docstring was listed; both now are.
5. **`operations.md` teaches the degraded-check description for `power_data_is_fresh` only.**
   Verified at `:169-176`. Added as a second edit to that page.
6. **Tests 1 and 2 fail on `main` for a different mechanism than stated** — `materialize()` defaults
   to `raise_on_error=True`, so the test errors rather than asserting `success is False`. The
   conclusion (they fail on `main`) is unchanged; the wording was.
7. **"Nothing can fail its own step" was overstated** — Dagster's own event serialisation and
   process death remain outside every guard. Carve-out added, with the note that the residual costs
   more here than it does for a standalone check.
8. **"Subsumed by #476" was the wrong disposal.** #476 owns the *replace* path, not "should
   `write_nwp` refuse to append onto an existing partition". Closing #509 on the original wording
   would have left that untracked. The action is now: add it to #476 explicitly.
9. **Departure 3 was framed as a correction the issue had partly already made** — the issue's own
   parenthetical scopes the retry ladder correctly. Softened to a clarification, with a note not to
   claim otherwise in the PR body.
10. **Minor:** `assess_nwp_run_completeness` also takes `expected_n_h3_cells`, so "the in-memory
    frame as their only argument" was wrong; and the identical-report argument is stronger than
    stated, because `write_nwp` never mutates its argument at all. Both corrected.

### Considered and rejected

- **`report_check_degradation`'s own guard is `except Exception`, narrower than the
  `except BaseException` guard that calls it** — so a `BaseException` out of `sentry_sdk` would
  escape into the caller's handler and fail the run. **Rejected for this issue, not on the merits:**
  it is pre-existing, affects the two existing callers identically, and is out of #509's scope
  (CLAUDE.md: report design mistakes outside the issue's scope rather than fixing them). Flagged to
  Jack in the report below; worth its own one-line issue.
- **Nothing was rejected as factually wrong.** The review found no false finding this time, which is
  itself worth recording — the usual expectation is that some are.
