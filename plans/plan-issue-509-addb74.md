# Plan — issue #509: `ecmwf_ens`'s in-asset checks run after the Delta write

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/509>
Branch: `claude/plan-issue-509-addb74`

## Verdict

**Worth implementing.** The bug is real and the fix is small: everything that computes and packages
the two WARN checks currently runs *after* `write_nwp`
([`assets.py:288-310`](../src/nged_substation_forecast/defs/assets.py)), so any raise in there fails
the run with the append already committed, and the operator's re-materialisation appends the run a
second time.

Two departures from the issue body:

1. **The write-idempotency half is not settled here.** The issue calls "what should the write do
   about a partition that already exists" the more important half. That is
   [#476](https://github.com/openclimatefix/nged-substation-forecast/issues/476)'s scope, and
   [`operations.md:223-230`](../docs/live_service/operations.md) already carries the operator rule
   pointing there. The PR body says "the write-side half is #476's" and that is the whole of it.
   The one thing worth adding to #476 as a comment: it currently plans only the *replace* path, not
   "should `write_nwp` refuse to append onto an existing partition" — one line, not a procedure.
2. **The retry framing is looser than the issue's title suggests.** `RetryRequested` is raised only
   inside the pre-write `try` (`assets.py:279-283`), there is no `RetryPolicy` on the asset or job,
   and no `run_retries` in either documented `dagster.yaml`. So duplication needs a *manual*
   re-materialisation of the red partition — which is exactly what an operator does, so the hazard
   stands. One sentence of PR-body wording; the issue's own parenthetical already said as much.

## What changes

### `src/nged_substation_forecast/defs/assets.py`

**The reorder is the fix.** Move the whole post-write block above `write_nwp`: both `assess_*`
calls, **both `AssetCheckResult` builders, and `_nwp_run_shape_metadata`**. Below the write, leave
only the log line and `return MaterializeResult(...)` over already-computed locals.

Moving only the two `assess_*` calls would be a botched fix, and this is the trap worth naming:
`_nwp_quality_check_result` calls `_nwp_null_slices_metadata` (`assets.py:404+`), which sorts,
`.head(100)`s and builds `TableRecord`s row by row over dict lookups and `str()` conversions. That
is by a distance the most raise-prone code in the block — far more so than the two pure
aggregations. Leave it below the write and the hazard is barely reduced.

Add one comment at the write saying why the order matters (`write_nwp` is `mode="append"` with no
dedup, [`delta_store/nwp.py:98-105`](../packages/delta_store/src/delta_store/nwp.py)), so the next
person does not tidy the assessments back underneath it.

**Plus one inline `try`/`except` around the moved block** — see "Why the guard stays" below for why
this is not gold-plating, and why it is *one inline guard*, not two extracted helpers. On the
degraded path:

- Re-raise `KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError`, log the traceback
  via `context.log.exception` (confirmed present on the pinned Dagster 1.13.17), call
  `report_check_degradation` once per check name, and return a `passed=False`,
  `AssetCheckSeverity.WARN` result for **each** check with `{exc!r}` in the description.
- **Both fallback results must set `check_name` explicitly.** This is the one trap that would
  silently reintroduce the failure being fixed: the fallback this pattern is copied from,
  `power_data_is_fresh`'s at [`checks.py:360-366`](../src/nged_substation_forecast/defs/checks.py),
  passes **no** `check_name`, because a standalone `@asset_check` does not need one. Verified by
  execution on the pinned Dagster: inside an asset declaring two `check_specs`, both an omitted
  result *and* a result with no `check_name` fail the step outright
  (`DagsterInvariantViolationError`, `success=False`, zero evaluations recorded).
- Shape metadata is simply absent (`{}`) — there is no `NwpRunCompletenessReport` to read it from.
  No sentinel values: `_nwp_run_shape_metadata`'s existing comment warns that a key whose metadata
  *type* changes between runs breaks the Dagster UI timeline plot. `n_rows`, `path` and `init_time`
  are computed outside the guard and always published.
- `_nwp_run_shape_metadata`'s docstring says the shape is "Published on *every* materialisation"
  (`assets.py:313-319`). That becomes false — fix the wording.

### Why the guard stays (and why it is the *only* thing added)

An earlier simplicity review argued for dropping the guard entirely: both `assess_nwp_quality` and
`assess_nwp_run_completeness` are documented as never raising on a validated `Nwp` frame, that was
verified true, and after the reorder a raise costs a red run *before* any write — loud, paged, and
recoverable by re-running. That is a good argument and it is why everything *else* it proposed
cutting has been cut.

It does not carry, because rule 7 applies independently of the duplication. Its own text names the
harm: a warning-path bug converts fail-open into fail-closed "at exactly the wrong moment", and for
`ecmwf_ens` the wrong moment is concrete — the NWP run does not land, so the live forecast runs on
yesterday's NWP, purely because a *reporting* function had a bug. The repo has already paid to fix
this exact pattern once, in #480. Closing #509 — an issue whose body is explicitly about rule 7 —
having fixed only the duplication would leave the rule-7 half open.

But the guard's *cost* is one inline `try` in the asset body, not the structure originally planned.
Rejected as gold-plating: two extracted helpers for one caller each (one of which had to return an
awkward `tuple[AssetCheckResult, dict[str, MetadataValue]]`), and splitting them so "a bug in one
does not blind the other" — both degraded results would carry the same `repr(exc)` and produce the
same Sentry event, so the split buys one check staying green and costs a whole extra function.
`_check_power_data_freshness` was extracted because its body is twenty lines; this one is five.

## Design-philosophy check

- **Production path, so degrade.** `ecmwf_ens` is a scheduled production ingest with the
  `sentry_capture_failure` hook. This is
  [rule 7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules);
  both checks stay `AssetCheckSeverity.WARN` / `blocking=False`, and after this change nothing
  either check's *body* does can fail its own step. State it with that carve-out, not absolutely:
  Dagster's own event serialisation and process death stay outside every guard, as `checks.py`'s
  module docstring already says.
- **Hypotheses.** [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself),
  measured by **T1.1** — a warning-path bug that duplicates NWP rows is an intervention whose
  recovery is a data repair rather than a re-run.
- Nothing is traded away: every non-degraded path materialises exactly as before, with the same
  results.

## Tests

Two, both in `tests/test_assets.py`'s `--- ecmwf_ens ---` section, reusing `_make_nwp` (`:111`),
`_write_h3_grid_weights` (`:125`), `env` (`:133`) and `_check_evaluations` (`:258`).

1. **`test_ecmwf_ens_assesses_before_writing`** — the ordering test, which is what pins *this
   issue's* fix. Patch `assets.assess_nwp_quality` with a recorder that appends
   `Path(Settings().nwp_data_path).exists()` and then delegates to the real function; materialise;
   assert the recorded list is `[False]`.
   **Fails on `main`:** the assessment runs after the write today, so the recorded value is `True`.
   *Why this shape:* it asserts the invariant directly ("nothing is on disk when the warning path
   runs") and works identically with or without the guard. The originally-planned version — raising
   `DagsterExecutionInterruptedError` so it escapes the guard's re-raise carve-out — coupled the
   ordering assertion to that carve-out list, so editing the list would have silently stopped the
   test from testing ordering.

2. **`test_ecmwf_ens_lands_the_run_when_an_assessment_raises`** — patch
   `assets.assess_nwp_quality` to raise `RuntimeError`; materialise. Assert `result.success`;
   `pl.read_delta(...).height == 4` (written exactly once); and that **both** check names appear in
   `_check_evaluations(result)`, with `nwp_has_no_unexpected_nulls` `passed is False` and
   `severity == AssetCheckSeverity.WARN`. The "both names present" assertion is the one that catches
   the missing-`check_name` trap.
   **Fails on `main`:** the raise propagates. Note the mechanism, because the obvious phrasing is
   wrong: `materialize()` defaults to `raise_on_error=True`, so the exception escapes the
   `materialize(...)` call itself and the test **errors** — it never reaches `assert
   result.success`. Do not describe it in a PR body as "`result.success` is False".

A second copy of test 2 patching `assess_nwp_run_completeness` was cut: it would differ only in
which function is patched, and its extra assertion (the other check stayed green) existed solely to
justify the two-helper split that is itself cut.

Existing `ecmwf_ens` tests must keep passing unchanged — if one needs editing, that is a signal the
change altered behaviour it should not have. Nothing in `packages/delta_store/tests/test_nwp.py`
changes; `write_nwp` is untouched.

## Docs to update

Four one-line corrections of statements this change makes false, plus one durable addition. Written
to describe the present, per CLAUDE.md.

- **[`docs/design-philosophy/inherent-stability.md:182-188`](../docs/design-philosophy/inherent-stability.md)**
  — the durable output of this issue, and the only edit worth more than a correction. Rule 7 names
  only `power_data_is_fresh` and `live_forecasts_are_healthy`. Add the two in-asset `ecmwf_ens`
  checks, and **one clause** generalising what the issue taught: a warning path computed *inside* an
  asset must also run **before** that asset's non-idempotent write, not merely under a guard.
- **[`docs/architecture/production-deployment.md:180-182`](../docs/architecture/production-deployment.md)**
  — states outright that "The two NWP checks are computed inside the `ecmwf_ens` asset and have no
  catch-all, so a raise there does fail the run and the hook does see it." Becomes false.
- **[`src/nged_substation_forecast/_sentry.py`](../src/nged_substation_forecast/_sentry.py)** — the
  module docstring (`:10`) and `report_check_degradation`'s docstring (`:136`) both scope the
  covered fault to "a standalone `@asset_check`". Both become false; widen to any guarded warning
  path.
- **`_nwp_run_shape_metadata`'s docstring** (`assets.py:313-319`) — "Published on *every*
  materialisation" becomes false on the degraded path.
- **[`docs/live_service/operations.md`](../docs/live_service/operations.md)** — the page teaches the
  operator to read `power_data_is_fresh`'s degraded description at `:180-186`. Both NWP checks gain
  that state. **One sentence** in the NWP section, not a new passage.

Deliberately **not** edited:

- `docs/architecture/ecmwf-ens-known-issues.md:193` — "A run that fails ingest writes nothing
  (validation runs before the Delta append)". Today that is subtly *false*; the reorder makes it
  true as written. The change fixes the doc; the doc needs no edit.
- `defs/checks.py`'s module docstring and `docs/architecture/why-dagster-not-airflow.md:202` — both
  stay accurate. Listed only so the reviewer does not flag them as missed.
- No roadmap ship-time triage: #509 completes no roadmap item, carries no milestone, and nothing in
  `docs/` links it.

## Verification commands

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check && uv run pytest
```

```bash
uv run pytest tests/test_assets.py -k ecmwf_ens -v
```

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict
```

`--run-network` is not needed: nothing here touches the convention-sensitive ECMWF conversion path.
`mkdocs build --strict` is, because three `docs/` pages change.

## Risks and open questions

1. **Should `write_nwp` refuse to append onto a partition that already exists?**
   **Recommendation: not here.** #476 has to decide how a replace is triggered, and a hard refusal
   landed first would either be relaxed by #476 or block its legitimate path. Add the question to
   #476; do not implement it in #509.
2. **Does this make "a red `ecmwf_ens` partition means nothing landed" safe to state flatly in the
   runbook?** **No — do not state it flatly.** A process killed between the Delta commit and Dagster
   recording success leaves a red partition with rows on disk. The reorder makes the *code* safe; it
   does not make that claim true. Flagged because it is the simplification a reviewer will push for.

## Review history

Two fresh sub-agents reviewed this plan, neither told the reasoning behind it.

**Review 1 — correctness.** Confirmed all four load-bearing claims (assessments run post-write;
`write_nwp` is append-only with no dedup; nothing auto-retries; pre- and post-write assessment are
identical — in fact `write_nwp` never mutates its argument at all). Ten findings applied, of which
one was material: **the degraded `AssetCheckResult` must set `check_name` explicitly**, because the
fallback this pattern copies omits it and a verbatim copy would fail the step. Also caught that
three doc pages assert the checks have no catch-all, that `assets.py` has no module-level `logger`,
and that tests fail on `main` by *erroring* under `raise_on_error=True` rather than by asserting
`success is False`. Nothing was rejected as factually wrong. Rejected for **scope**, and flagged to
Jack instead: `report_check_degradation`'s own guard is `except Exception`, narrower than the
`except BaseException` guard that calls it, so a `BaseException` out of `sentry_sdk` would escape —
pre-existing, affects both current callers, worth its own issue.

**Review 2 — simplicity.** Cut roughly two-thirds of the plan. Accepted in full: two guarded helpers
→ one inline `try`; the degraded-metadata "policy" → `{}`; three tests → two, with a better ordering
test; five doc edits → four corrections plus one addition, dropping the
`ecmwf-ens-known-issues.md` edit (the reorder makes that sentence true rather than needing it
rewritten) and two non-edits; the two-issue comment procedure → one line; sixteen lines on the retry
ladder → one sentence. It also caught a doc edit the first review missed
(`_nwp_run_shape_metadata`'s "every materialisation"), and confirmed `DagsterLogManager.exception`
exists so the hedge about it could go.

**Rejected: its headline recommendation to drop the catch-all entirely.** Its evidence is right —
both assess functions really are non-raising, and after the reorder a raise is loud and safe — but
rule 7 applies independently of the duplication, and the harm it names is concrete here (a
reporting bug costs the live forecast a day of NWP freshness). #480 fixed this same pattern for
another check. Its own fallback position was the same single inline guard, and that is what the
plan now specifies. It also read
`production-deployment.md:180-182` as documenting the unguarded state as a deliberate design
*feature*; in context that parenthetical is a factual aside explaining why
`report_check_degradation` has two callers, not an argument for leaving the NWP checks unguarded.
