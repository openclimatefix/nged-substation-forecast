# Categorise Sentry events (#488)

**The problem.** A transient object-store error during the hourly `power_time_series_and_metadata`
run reaches Sentry looking exactly like a `TypeError` in our own code, even though no data is lost
and the next hourly run back-fills. Separately, when a run *does* fail after exhausting an in-band
retry, Sentry groups it under `RetryRequested` rather than under the exception that actually
happened — so the issue title says "something retried" instead of "the upstream NWP run never
published".

**The planned solution.** Four small changes, none of which needs a list of "upstream" exception
types to keep honest:

1. A short `RetryPolicy` on `power_time_series_and_metadata`, so a transient blip never becomes a
   Sentry event at all — the step is retried and the run succeeds.
2. `sentry_capture_failure` reports the underlying cause when Dagster hands it a `RetryRequested`,
   so an exhausted-retry failure groups under the real exception.
3. `sentry_capture_failure` sends nothing at all when the exception is a run cancellation. This is
   **not** optional polish: change 1 is what makes a cancelled run reach the hook in the first
   place (verified below), so without this guard the plan would make Sentry noisier, not quieter.
4. One `fault_category: run_failed` tag on the failure hook's events, giving the one class that
   should alert a human a *positive* marker to key an alert rule off, rather than an eliminative
   rule that silently misclassifies the day a fifth sender is added.

Every change lands in `src/nged_substation_forecast/_sentry.py`, except change 1, which is one
decorator argument in `defs/assets.py`. `defs/checks.py` is untouched, and no public signature
changes.

## Verdict, size and departures

**Verdict: worth implementing, and the issue comment's ordering is right.** Its weak preference —
retry first, categorise only what still gets through — is what this plan does. The retry is the
larger win because it removes the commonest false page rather than labelling it.

**Size: complex.** It changes the production failure behaviour of the hourly ingest job, which is a
degradation rule under `docs/design-philosophy/inherent-stability.md`, and more than one design
defensibly satisfies the issue. It gets this plan, both plan reviews, and both diff reviews in
`implement-issue`.

**Departures from the issue's comment:**

- **The comment's second remedy — "tag the event by exception type … it needs a list of 'upstream'
  exception types that will drift" — is rejected outright, not deferred.** The list genuinely cannot
  be kept honest: each compiled extension the ingest path reads through (obstore, delta-rs, polars)
  defines its own exception classes, and a pyo3 panic surfaces as a `PanicException` with no stable
  identity — the same argument the existing `except BaseException` guard in
  `checks.py::power_data_is_fresh` already makes in a comment. Sentry also already groups issues by
  exception type and puts it in the title, so a tag restating it adds nothing; and once change 1 is
  in place, the dominant upstream case never reaches Sentry at all. A design that *would* work — a
  boundary-declared marker exception — is in "Risks and open questions" as a possible follow-up.
- **The comment says class 3 (`report_check_degradation`) already has "the `asset_check` tag there
  to key an alert rule off". Agreed — so class 3 gets no code change**, only a docs line. The same
  goes for class 2's `degraded_asset` tag and for the freshness warning's `level="warning"` plus
  per-environment fingerprint. Only the failure hook, which carries no distinguishing mark today,
  gains a tag.
- **The plan adds two changes the issue does not mention** (changes 2 and 3). Both are
  categorisation defects of exactly the kind the issue is about, both are verified live behaviour
  rather than theory, and together they are about six lines.

## Verified facts this design rests on

Established by probing the installed Dagster 1.13.17 and reading its source, not from documentation.

1. **A `failure_hook` does not fire on an attempt that will be retried.** An asset with
   `RetryPolicy(max_retries=2)` that failed twice then succeeded ran 3 attempts and fired the hook
   **0 times**; the run succeeded. A step going up for retry emits `STEP_UP_FOR_RETRY`, and
   `failure_hook` tests `event.is_step_failure`, which is `STEP_FAILURE` only
   (`hook_decorator.py:259`, `events/__init__.py:660`). This is what makes change 1 *remove* the
   event rather than multiply it.
2. **After the retries are exhausted the hook fires once, seeing the original exception.** An
   always-failing asset with the same policy ran 3 attempts, fired the hook once, and the hook saw
   `TypeError`. The mechanism is upstream: `HookContext.op_exception` already unwraps
   `RetryRequestedFromPolicy` to its `__cause__` (`_core/execution/context/hook.py:130-137`). So
   Sentry grouping by real exception type survives change 1 untouched.
3. **That upstream unwrap covers only `RetryRequestedFromPolicy`, not a plain `RetryRequested`.** A
   hand-raised `RetryRequested` reaches the hook as itself, with the real exception on `__cause__`.
   This is live today for `ecmwf_ens`: a run that never publishes exhausts its 8 retries
   (`assets.py:369-371`) and reaches Sentry grouped as `RetryRequested`, hiding
   `NwpRunNotYetAvailable`. Change 2 extends Dagster's own unwrap one class up the hierarchy.
4. **A `RetryPolicy` converts an interrupt into a bare `RetryRequested`** —
   `_core/execution/plan/utils.py:94-102` catches `(DagsterExecutionInterruptedError,
   KeyboardInterrupt)` and re-raises as `RetryRequested` when a policy is set, commented "respect
   retry policy when interrupts occur".
5. **Consequently, change 1 would newly report every cancelled run to Sentry.** Probed both ways
   against a job carrying a failure hook: **no policy → the hook fires 0 times** (the interrupt path
   at `execute_plan.py:319-326` yields a step-failure event and then re-raises, so `_trigger_hook`
   at `execute_plan.py:99` is never reached); **with a policy → the hook fires once, seeing
   `RetryRequested`** (the interrupt has become a retry request, which exhausts its budget and
   yields a step failure *without* re-raising). Change 3 exists for this. Note the interaction with
   change 2: without change 3, the unwrap would faithfully report the cancellation as
   `DagsterExecutionInterruptedError` — correctly labelled and still unwanted.
6. **Dagster's exponential backoff is `delay × (2ⁿ − 1)`, not `delay × 2ⁿ⁻¹`.** Measured:
   `RetryPolicy(max_retries=3, delay=2, backoff=EXPONENTIAL)` yields waits of 2 s, 6 s, 14 s —
   22 s total, not the 14 s a doubling series would suggest. The budget below is computed from the
   measured figures.
7. **Retries are enabled under `materialize()` / `execute_in_process`**, so the existing tests in
   `tests/test_assets.py` will exercise the new policy rather than silently skipping it. And
   `power_time_series_and_metadata.op.retry_policy` is readable from a test (it is `None` today).

## What changes, file by file

### `src/nged_substation_forecast/defs/assets.py` — one decorator argument

Line 75 becomes:

```text
@asset(tags=PRODUCTION_LAYER_TAGS, retry_policy=_POWER_INGEST_RETRY_POLICY)
```

with `_POWER_INGEST_RETRY_POLICY: Final = RetryPolicy(max_retries=2, delay=1,
backoff=Backoff.EXPONENTIAL)` just above the asset, carrying a docstring stating the budget and why
it is short. `RetryPolicy` and `Backoff` join the existing `dagster` import block.

**The budget is 1 s + 3 s = 4 s of waiting across 3 attempts, and is deliberately short.** The whole
value of the retry is suppressing an alert; the data is never at risk, because the asset is
unpartitioned, re-lists NGED's S3 bucket from scratch each attempt, and the next hourly run
back-fills whatever this one missed. So the budget only has to cover an *instantaneous* glitch, not
an outage. A longer budget would buy nothing the next hour does not already buy, while making a
cancelled run slower to stop (fact 4) and making every retried attempt re-do the S3 listing. A
persistent outage still fails after 4 s and alerts, which is the wanted behaviour. This is rule 10
of `inherent-stability.md` — bounded retries with backoff — applied at its cheapest useful setting.

No `jitter`: jitter de-synchronises many workers retrying in lockstep, and this is one hourly job.

**Retrying is safe.** The asset is idempotent across attempts: it re-lists the bucket,
`select_new_rows` dedupes the power frame against the Delta table before appending, and
`upsert_metadata` is an upsert. The Delta append is the last statement in the body, so a retry
cannot double-write. The new test below asserts this rather than taking it on trust.

**Nothing else in `assets.py` is touched** — not the `report_asset_degradation` call at :146, and
nothing in the NWP region. #506's and #580's territory is untouched.

### `src/nged_substation_forecast/_sentry.py` — three changes, all inside `sentry_capture_failure`

The function currently captures `context.op_exception` whenever it is not `None`
(`_sentry.py:119-132`). It gains, in order:

1. **The `RetryRequested` unwrap.** When the exception is a `dagster.RetryRequested` with a
   non-`None` `__cause__`, work with the cause instead. A code comment should cite
   `HookContext.op_exception`'s own unwrap at `hook.py:134` and say this extends it to the plain
   class, which `assets.py:369` raises. Must tolerate `__cause__ is None` without crashing.
2. **The cancellation guard.** If the resulting exception is a
   `KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError`, return without capturing. An
   operator cancelling a run is not a fault. The comment must say *why* it is needed — fact 5, that
   the retry policy is what routes cancellations here — because with the policy removed the guard
   would look like dead code and get deleted. Applying the guard *after* the unwrap is what catches
   the real case, since the cancellation arrives wrapped.
3. **The `fault_category: run_failed` tag**, set on a forked `sentry_sdk.new_scope()` so it cannot
   leak into later events, with the value as a `Final` module constant.

**Why only this sender gets a tag.** The other three already carry a distinguishing mark:
`report_asset_degradation` tags `degraded_asset` (:169-174), `report_check_degradation` tags
`asset_check` (:197-202), and the freshness warning is `level="warning"` with a stable
per-environment fingerprint (:316, :332). Adding a fourth value to each would re-encode what is
already there. The failure hook is the one sender with nothing, and it is the one class that should
alert a human — so it is the one that must not be identified by elimination. A rule of the form
"`level:error` and `asset_check` is not set and `degraded_asset` is not set" works today and
silently misclassifies the day a fifth sender is added; a positive marker does not.

**Docstrings.** The module docstring and `init_sentry`'s both describe the senders; update them for
the tag and the two new guards. `init_sentry`'s claim that exactly four explicit senders reach
Sentry stays true — no fifth sender is added.

### Docs

Three pages, one passage each. **`docs/design-philosophy/inherent-stability.md` is not edited** —
this change applies its existing rules rather than adding one.

- **`docs/architecture/production-deployment.md`**, § "Send telemetry to Sentry, and alarm on
  absence" (~lines 194–260): one paragraph on the ingest retry policy (a blip is not an event; the
  budget is short because the next hourly run back-fills), one sentence that a cancelled run is
  deliberately not reported, and one recording that we do *not* classify upstream-vs-ours by
  exception type, with the drift argument.
- **`docs/live_service/sentry.md`**: a fourth console step under "Turn it on in production" giving
  the alert-rule recipe — alert on `fault_category:run_failed`; route `asset_check:*`,
  `degraded_asset:*` and `level:warning` to a non-alerting channel. Phrase urgency in the vocabulary
  `inherent-stability.md` already uses ("next business day"), **not** as a page: that page's
  failure-mode table uses only "No" and "Yes, next business day", and line 153 states "Nothing here
  is a 2am page."
- **`docs/live_service/operations.md`**: beside the existing `asset_check:…` / `degraded_asset:…`
  runbook mentions (~lines 227, 234, 306), one clause noting that a transient object-store error on
  the hourly ingest is retried and no longer reports, and one noting that because a retry re-runs
  the whole asset body, a degradation event from `upsert_metadata` (`assets.py:136-147`) can appear
  more than once for a single run.

## Design-philosophy check

This path is **production**, so it must degrade rather than raise, and it does:

- The retry policy strictly *reduces* production failures; it cannot introduce one. A persistent
  fault still fails after 4 s exactly as today.
- **No warning path gains the ability to raise.** Neither `_capture_tagged` nor
  `report_power_freshness` is touched, so their existing `try`/`except` guards are unchanged.
  `sentry_capture_failure` gains a `new_scope()` block; it runs inside Dagster's hook error boundary
  (a raising hook yields `HOOK_ERRORED`, not a run failure — `execute_plan.py:146`) and is only
  reached on a step that has already failed, so it cannot turn fail-open into fail-closed. It stays
  a no-op with an empty DSN because `capture_exception` needs an active client.
- **Liberal about missing inputs, strict about malformed ones**: a transient object-store error is
  the outside world misbehaving, so retrying and carrying on is the required response.
- No asset check is added or changed, so the `WARN`/`blocking=False` rule is not engaged.
- Rule 10 (bounded retries with backoff) is the rule this implements; cite the exact label from
  `engineering-hypotheses.md` when the docs edit lands rather than guessing at it here, since the
  labels are append-only and must not be misapplied.

Against `design-principles.md`, nothing is traded away. The one debit is fact 4: a cancelled run now
takes up to 4 s of retried attempts to stop instead of stopping at once. That is the price of using
Dagster's own retry mechanism, and the short delay is chosen to bound it.

## Tests

Every assertion below fails on `main` today.

**`tests/test_sentry.py`** — all cheap unit tests of the hook, using the existing scope-stubbing
machinery in `test_degradation_reporters_capture_the_exception_and_tag_the_name` (:193):

1. `test_failure_hook_reports_the_cause_of_a_retry_requested` — a `RetryRequested` whose `__cause__`
   is an `OSError`; assert the captured exception **is** the `OSError`. *Fails on `main`*: today the
   `RetryRequested` itself is captured.
2. `test_failure_hook_captures_a_retry_requested_with_no_cause` — the same class with `__cause__`
   unset is captured as itself. Guards the unwrap against a `None` dereference; it would fail
   against a naive implementation of change 2 rather than against `main`, and is worth keeping for
   that reason.
3. `test_failure_hook_ignores_a_cancelled_run` — parametrised over a bare
   `DagsterExecutionInterruptedError`, a `KeyboardInterrupt`, and a `RetryRequested` wrapping one
   (the case fact 5 shows is the reachable one); assert `capture_exception` is never called. *Fails
   on `main`*: all three are captured today.
4. `test_failure_hook_tags_the_fault_category_without_leaking_scope` — assert the tag value and that
   it is set on a forked scope, modelled on `test_report_power_freshness_does_not_leak_scope` (:388).
   *Fails on `main`*: the hook sets no tag and forks no scope.

**`tests/test_assets.py`**, in the `test_power_time_series_and_metadata_*` block (:195–375 — this
session's territory; #580 owns only the `test_ecmwf_ens_*` block):

1. `test_power_time_series_and_metadata_retries_a_transient_failure` — stub
   `list_timeseries_json_files` to raise `OSError` on its first call and delegate to the real
   listing afterwards; `materialize(...)`; assert the run succeeded, that the stub was called twice,
   **and that the resulting power table and metadata parquet are exactly what the happy-path test
   asserts**. That last clause is the point: it covers the idempotency claim above, which is ours,
   rather than merely covering Dagster's retry machinery. Costs ~1 s (the first retry delay).
2. `test_power_time_series_and_metadata_declares_its_retry_budget` — assert
   `power_time_series_and_metadata.op.retry_policy` has the intended `max_retries`, `delay` and
   `backoff`. Costs nothing, and catches the budget being changed or the argument dropped — the one
   regression the behavioural test above cannot distinguish. Precedent for asserting wiring at
   definition level: `test_failure_hook_is_attached_to_the_scheduled_jobs` (`test_sentry.py:255`).

**Deliberately not written**: a test that exhausts the retry budget and asserts the run fails. It
would sleep for the whole 4 s budget to prove a property of Dagster rather than of this repo, which
test 2 already pins down for free.

**Existing test to re-check, not change**:
`test_power_time_series_and_metadata_re_raises_a_cancelled_run` still passes (the run still ends
unsuccessful) but now costs ~4 s of retried attempts. Measure it; if it is a problem, the recorded
fallback is to monkeypatch that one op's retry policy.

## Verification commands

The green-before-push set, from the worktree root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

`mkdocs build --strict` matters because the docs changes add cross-page links between
`production-deployment.md`, `sentry.md` and `operations.md`. No network-marked tests are implicated.
Time `uv run pytest` before and after and report both numbers in the PR — the retry policy adds
wall-clock to two tests by design.

## Risks and open questions

1. **A `RetryPolicy` makes a cancelled run take ~4 s to stop, and routes it to the failure hook**
   (facts 4 and 5, both verified). *Recommendation: accept, with change 3 as the mitigation.* The
   alternative is an in-band `RetryRequested` wrapped in the repo's `except BaseException` /
   re-raise-interrupts idiom, which keeps cancellation instant — but it costs a second guard block
   in the asset body, covers only the region it wraps, and hands the hook a `RetryRequested`, which
   is the very defect change 2 exists to fix.
2. **Is a 4 s retry budget too short to be worth having?** *Recommendation: no — it is the right
   size for the job it has*, per the assets.py section: the hourly back-fill, not the retry, is what
   protects the data. If the reviewer wants minutes instead, raise `delay` and drop the behavioural
   test rather than letting it sleep.
3. **Should blame — upstream vs our code — be tagged at all?** This plan says no, and says why an
   exception-type allowlist cannot be kept honest. A design that *would* work is a project-defined
   `UpstreamDataError` raised at each boundary where we call the outside world (the NGED S3 read in
   `packages/nged_data/storage.py`, the Dynamical.org read in `packages/dynamical_data/`), with the
   hook walking `__cause__` and tagging `fault_source` — it cannot drift, because each boundary
   declares its own status locally. *Recommendation: not now.* It touches two packages, one of which
   is #506's territory, and its value drops sharply once change 1 removes the common case. Worth its
   own issue if Sentry triage still feels slow after this ships.
4. **Should `live_forecasts` and `ecmwf_ens` get retry policies too?** Out of scope. `ecmwf_ens`
   already has an in-band retry for the condition that warrants one, and `live_forecasts` is #528's
   territory. *Recommendation: leave both; open a separate issue if their events prove noisy.*

## Findings from review 1 (simplicity) that were rejected

- **"Cut `fault_category` entirely; an alert rule can identify the paging class by elimination
  (`level:error` and `asset_check` not set and `degraded_asset` not set)."** Rejected for the one
  sender that alerts a human, accepted for the other three. An eliminative rule is correct today and
  fails silently the day a fifth sender is added, and it is the alerting rule — the one place that
  fragility is least affordable. Cost of the positive marker: about five lines. The finding's
  stronger half — that the other three senders are already distinguishable — **is** adopted, and cut
  three quarters of the original proposal.
- **"Use Sentry's native `level` instead of a new tag."** Rejected: `level:warning` already means
  "upstream telemetry is late" on the freshness event, so reusing it for check degradation would
  merge two unrelated meanings onto one facet. Moot in any case now that only one sender is tagged.
- **"An operator-cancelled run sends a Sentry error event today, so the interrupt guard fixes a
  pre-existing noise class."** Rejected as stated, and adopted with the opposite polarity. Probing
  showed the hook fires **0 times** on cancellation today, because the interrupt path re-raises
  before `_trigger_hook` is reached. The guard is not a pre-existing fix; it is a mandatory
  mitigation for the regression change 1 would otherwise introduce (fact 5). Same code, different
  reason — and the reason matters, because a comment claiming it fixes today's behaviour would be
  false and would invite deletion.
- **"Drop the behavioural retry test; assert the definition only."** Rejected in part. The
  budget-exhaustion test is dropped, as proposed. The transient-failure test is kept, because at the
  corrected 1 s cost it also covers the idempotency-across-retries claim, which is a property of
  this repo's asset and not of Dagster — a gap the finding itself identified.

Accepted in full from the same review: the backoff arithmetic error (fact 6, the original budget was
out by 57%), the reduced `max_retries=2, delay=1` budget, citing `hook.py:134` for the unwrap, the
runbook clause about a retry re-emitting a degradation event, and trimming the docs surface to three
passages with `inherent-stability.md` left alone.
