# Categorise Sentry events (#488)

**The problem.** Every Sentry event this repo sends arrives without a machine-readable statement of
how urgent it is or what kind of fault it represents. A transient object-store error during the
hourly `power_time_series_and_metadata` run pages exactly like a `TypeError` in our own code, even
though no data is lost and the next hourly run back-fills. At the other end, an asset check that
could not evaluate its own inputs — the service keeps forecasting throughout — arrives looking just
as urgent as a failed production run.

**The planned solution.** Two changes, both small, in that order. First, a short `RetryPolicy` on
`power_time_series_and_metadata`, so a transient object-store blip never becomes a Sentry event at
all: the step is retried and the run succeeds. Second, one `fault_category` tag with a closed
four-value vocabulary, set on every event the four senders in `_sentry.py` emit, so an alert rule
can route by urgency. A third, one-line change makes `sentry_capture_failure` report the underlying
cause when Dagster hands it a `RetryRequested`, so a run that exhausts its retries groups in Sentry
under the real exception rather than under `RetryRequested`.

The whole categorisation lands inside `src/nged_substation_forecast/_sentry.py`. `defs/checks.py`
is untouched, and `defs/assets.py` changes by exactly one decorator argument.

## Verdict, size and departures

**Verdict: worth implementing, roughly as the issue's comment describes.** The comment's weak
preference — retry first, categorise only what still gets through — is right, and this plan adopts
it. The retry is the larger win: it removes the single commonest false page rather than labelling
it.

**Size: complex.** It changes the production failure behaviour of the hourly ingest job, which is a
degradation rule under `docs/design-philosophy/inherent-stability.md`, and more than one design
defensibly satisfies the issue (retry vs categorise; `RetryPolicy` vs in-band `RetryRequested`; tag
vs level vs fingerprint vs exception allowlist). It therefore gets this plan, both plan reviews, and
both diff reviews in `implement-issue`.

**Departures from the issue's comment:**

- **The comment's second candidate remedy — "tag the event by exception type … it needs a list of
  'upstream' exception types that will drift" — is rejected outright, not merely deferred.** Three
  reasons. The list genuinely cannot be kept honest: each compiled extension the ingest path reads
  through (obstore, delta-rs, polars) defines its own exception classes, and a pyo3 panic surfaces
  as a `PanicException` with no stable identity — this is the same argument the existing
  `except BaseException` guard in `checks.py::power_data_is_fresh` already makes in a comment.
  Second, Sentry already groups issues by exception type and puts it in the issue title, so a tag
  restating it adds nothing. Third, once the retry policy is in place the dominant upstream case
  never reaches Sentry, so the list would be maintained for a rare residue. A design that *would*
  work — having the boundary declare upstreamness by raising a project-defined marker exception —
  is recorded under "Risks and open questions" as a possible follow-up, not planned here.
- **The comment says class 3 (`report_check_degradation`) has "the `asset_check` tag already there
  to key an alert rule off". That is true but not sufficient**, so class 3 still gets a one-word
  change. Keying a rule off "tag `asset_check` is set" works, but it is a rule per mechanism rather
  than a rule per urgency, it cannot be grouped or faceted in the Sentry issue stream, and it
  leaves the two loudest senders (`sentry_capture_failure`, `report_power_freshness`) with no
  category at all. One tag with a value on *every* event is the thing an alert rule and a saved
  search both want.
- **The plan adds a third change the issue does not mention** — unwrapping `RetryRequested` in the
  failure hook. It is in scope because it is a categorisation defect of exactly the kind the issue
  is about, it is verified to be live today (see below), and it is one `if` statement.

## Verified facts this design rests on

All four were established by running probes against the installed Dagster 1.13.17, not by reading
documentation. The probe scripts are throwaway; the findings are what matters.

1. **A `failure_hook` does not fire on an attempt that will be retried.** An asset with
   `RetryPolicy(max_retries=2)` that fails twice then succeeds ran 3 attempts and fired the hook
   **0 times**; the run succeeded. This is what makes the retry policy *remove* the event rather
   than multiply it — a step going up for retry emits `STEP_UP_FOR_RETRY`, and `failure_hook` tests
   `event.is_step_failure`, which is `STEP_FAILURE` only
   (`dagster/_core/definitions/decorators/hook_decorator.py:259`,
   `dagster/_core/events/__init__.py:660`).
2. **After the retries are exhausted the hook fires exactly once, and `context.op_exception` is the
   original exception.** An always-failing asset with the same policy ran 3 attempts, fired the hook
   once, and the hook saw `TypeError` — not `RetryRequestedFromPolicy`. So Sentry grouping by real
   exception type survives the retry policy untouched.
3. **In-band `RetryRequested` is different: the hook sees `RetryRequested`, with the real exception
   on `__cause__`.** This is live today for `ecmwf_ens`: a run that never publishes exhausts its 8
   retries and reaches Sentry grouped as `RetryRequested`, hiding `NwpRunNotYetAvailable`. Change 3
   fixes it.
4. **A `RetryPolicy` converts an interrupt into a retry request.** An asset raising
   `DagsterExecutionInterruptedError` ran 4 attempts with `max_retries=3` against 1 attempt with no
   policy — `dagster/_core/execution/plan/utils.py:94-102` catches
   `(DagsterExecutionInterruptedError, KeyboardInterrupt)` and re-raises as `RetryRequested` when a
   policy is set, commented "respect retry policy when interrupts occur". The run still ends
   unsuccessful. This is the one genuine cost of the chosen mechanism and it drives the delay budget
   below.

Also confirmed: retries **are** enabled under `materialize()` / `execute_in_process`, so the
existing tests in `tests/test_assets.py` will exercise the new policy rather than silently skipping
it.

## What changes, file by file

### `src/nged_substation_forecast/defs/assets.py` — one decorator, nothing else

Line 75 becomes:

```text
@asset(tags=PRODUCTION_LAYER_TAGS, retry_policy=_POWER_INGEST_RETRY_POLICY)
```

with `_POWER_INGEST_RETRY_POLICY: Final = RetryPolicy(max_retries=3, delay=2,
backoff=Backoff.EXPONENTIAL)` defined just above the asset, carrying a docstring that states the
budget and why it is short. `RetryPolicy`, `Backoff` join the existing `dagster` import block.

**Why the budget is 2 s + 4 s + 8 s = 14 s and deliberately not longer.** The entire value of the
retry is suppressing a page; the data itself is never at risk, because the run is unpartitioned, it
re-lists NGED's S3 bucket from scratch each time, and the next hourly run back-fills whatever this
one missed. So the budget only has to cover an *instantaneous* glitch, not an outage — a longer
budget would buy nothing the next hour does not already buy, while making a cancelled run slower to
stop (fact 4) and every retried attempt re-do the S3 listing. A persistent outage still fails after
14 s and pages, which is the wanted behaviour.

No `jitter`: jitter exists to de-synchronise many workers retrying in lockstep, and this is one
hourly job.

**Retrying is safe.** The asset is idempotent across attempts: it re-lists the bucket,
`select_new_rows` dedupes the power frame against what is already on the Delta table before
appending, and `upsert_metadata` is an upsert. The Delta append is the last statement in the body,
so a retry cannot double-write.

**The `report_asset_degradation` call at line 146 does not change** — its `fault_category` tag is
set inside `_sentry.py`. Nothing else in `assets.py` is touched, so #506's NWP region and #580's
`ecmwf_ens` docstring are untouched.

### `src/nged_substation_forecast/_sentry.py` — the category tag and the `RetryRequested` unwrap

**A closed vocabulary**, as a `Final` module constant per value (matching the repo's `Final`
convention) plus the tag key:

| Tag value | Sender | What it means | Operator urgency |
|---|---|---|---|
| `run_failed` | `sentry_capture_failure` | A scheduled production job failed. That cycle did not run. | Page. |
| `asset_degraded` | `report_asset_degradation` | The asset caught its own exception and carried on with reduced function. Data still landed. | Same working day. |
| `check_degraded` | `report_check_degradation` | A warning path could not evaluate its inputs. Forecasting is unaffected; we are blind to one signal. | Lowest — next working day. |
| `data_stale` | `report_power_freshness` | Upstream telemetry is late. Already `warning`-level and fingerprinted per environment. | Per the existing freshness rule. |

Changes:

- `_capture_tagged` gains a `fault_category: str` parameter and sets it as a second tag on the same
  forked scope. Its two callers, `report_asset_degradation` and `report_check_degradation`, pass
  their constant. **Neither of their public signatures changes**, so the `report_check_degradation`
  call site at `defs/assets.py:409` (inside #506's region) and both call sites in `defs/checks.py`
  are untouched — the hazard flagged in the wave brief does not arise.
- `sentry_capture_failure` sets `fault_category: run_failed` on a forked scope. It must keep working
  when Sentry is uninitialised, and it must not leak the tag, so it uses the same
  `sentry_sdk.new_scope()` shape the other senders use.
- `_capture_power_freshness_warning` sets `fault_category: data_stale` alongside its existing
  `n_late` / `n_stale` / `n_never_reported` tags and its fingerprint.

**The `RetryRequested` unwrap**, in `sentry_capture_failure`: when `context.op_exception` is a
`dagster.RetryRequested` with a non-`None` `__cause__`, capture the cause instead. This is what
makes an exhausted-retry failure group in Sentry under the exception that actually happened. It
affects `ecmwf_ens` today (fact 3) and will affect `power_time_series_and_metadata` only under
`RetryPolicy`, where fact 2 shows the hook already receives the original exception — so the unwrap
is a no-op there and a fix for the in-band case.

**The module and `init_sentry` docstrings** are updated to describe the category tag. The
`init_sentry` docstring's claim that exactly four explicit senders reach Sentry stays true: no
fifth sender is added.

### Docs

- **`docs/architecture/production-deployment.md`**, § "Send telemetry to Sentry, and alarm on
  absence" (around lines 194–260): add the four-value `fault_category` table above, the reasoning
  for the ingest retry policy (a blip is not an event; the budget is short because the next hourly
  run back-fills), and one paragraph recording that we deliberately do *not* classify upstream-vs-ours
  by exception type, with the drift argument.
- **`docs/live_service/sentry.md`**: add a fourth console step under "Turn it on in production"
  giving the alert-rule recipe — page on `fault_category:run_failed`, route
  `fault_category:asset_degraded` and `fault_category:check_degraded` to a non-paging channel — and
  mention the tag in the laptop verification section so a developer can see it in the UI.
- **`docs/live_service/operations.md`**: the runbook already names the `asset_check:…` and
  `degraded_asset:…` tags at lines ~227, ~234 and ~306; add the `fault_category` value beside each,
  and a line stating that a transient object-store error on the hourly ingest is retried and does
  not page.
- **`docs/design-philosophy/inherent-stability.md`**: read it during implementation and add a line
  only if the retry-before-degrade ordering leaves an existing rule inconsistent. Do not otherwise
  edit it — the change is an application of the existing rules, not a new one.

## Design-philosophy check

This code path is **production**, so it must degrade rather than raise. It does:

- The retry policy strictly *reduces* the number of production failures; it cannot introduce one.
  A persistent fault still fails after 14 s exactly as it does today.
- No warning path gains the ability to raise. The `fault_category` tag is set inside
  `_capture_tagged`, whose whole body is already wrapped in `try`/`except Exception` with an
  `logger.exception` fallback, and inside `_capture_power_freshness_warning`, which
  `report_power_freshness` already wraps for the same reason. `sentry_capture_failure` gains a
  `new_scope()` block; it runs in Dagster's hook error boundary (a raising hook yields
  `HOOK_ERRORED`, not a run failure — `execute_plan.py:146`), and it is only reached on a step that
  has already failed, so it cannot turn fail-open into fail-closed. It stays a no-op with an empty
  DSN because `capture_exception` needs an active client.
- **Liberal about missing inputs, strict about malformed ones**: a transient object-store error is
  the outside world misbehaving, so retrying and carrying on is the required response, not raising.
- No asset check is added or changed, so the `WARN`/`blocking=False` rule is not engaged.

Against `docs/design-philosophy/engineering-hypotheses.md`, this serves the operability claims
behind the always-output design: the alarm channel is only useful if what arrives on it is worth
waking for. Cite the specific label when the docs edit lands rather than guessing at it here — the
labels are append-only and must not be misapplied.

Against `docs/design-philosophy/design-principles.md`: nothing is traded away. The one debit is
fact 4 — a cancelled run now takes up to ~14 s of retried attempts to stop instead of stopping at
once. That is the price of using Dagster's own retry mechanism rather than hand-rolling one, and the
short delay is chosen to bound it.

## Tests

Every assertion below fails on `main` today.

**`tests/test_sentry.py`:**

1. `test_failure_hook_reports_the_cause_of_a_retry_requested` — build a `RetryRequested` whose
   `__cause__` is an `OSError`, pass it through the hook with `capture_exception` stubbed, assert
   the captured exception **is** the `OSError`. *Fails on `main`*: today the hook captures the
   `RetryRequested`.
2. `test_failure_hook_reports_the_exception_itself_when_it_is_not_a_retry_request` — the existing
   `test_failure_hook_captures_the_real_exception` covers the happy case; extend it or add a sibling
   asserting a `RetryRequested` with **no** `__cause__` is captured as itself, so the unwrap cannot
   turn into a crash on `None`. *Fails on `main`* only if the unwrap is written wrongly — this one
   is a guard for the new code rather than a test of a behaviour change, and is worth keeping for
   that reason.
3. `test_every_sender_sets_a_fault_category_tag` — parametrised over the four senders, driving each
   with a stubbed scope and asserting the `fault_category` tag value. *Fails on `main`*: no sender
   sets the tag. Model it on the existing `test_degradation_reporters_capture_the_exception_and_tag_the_name`,
   which already has the scope-stubbing machinery.
4. `test_failure_hook_does_not_leak_the_category_tag` — mirror of the existing
   `test_report_power_freshness_does_not_leak_scope`, asserting the hook's tag is set on a forked
   scope. *Fails on `main`*: the hook forks no scope today.

**`tests/test_assets.py`** (the `test_power_time_series_and_metadata_*` block, :195–375 — this
session's territory; #580 owns only the `test_ecmwf_ens_*` block):

1. `test_power_time_series_and_metadata_retries_a_transient_failure` — stub
   `list_timeseries_json_files` to raise `OSError` on its first call and delegate to the real
   listing afterwards; `materialize(...)`; assert `result.success` **and** that it was called twice.
   *Fails on `main`*: with no retry policy the first `OSError` fails the run. Costs ~2 s of wall
   clock (the first retry delay).
2. `test_power_time_series_and_metadata_gives_up_after_its_retry_budget` — stub the same function to
   raise every time; assert the run fails and the stub was called exactly `max_retries + 1` times.
   *Fails on `main`*: 1 call, not 4. This is the test that pins the budget down; it costs ~14 s, and
   if that proves unacceptable in the suite the fallback is to drop it and keep only a definition
   assertion (see the open question below).

**Existing test to re-check, not change**: `test_power_time_series_and_metadata_re_raises_a_cancelled_run`
still passes (fact 4 — the run still ends unsuccessful) but now costs ~14 s of retried attempts. If
the suite's total runtime becomes a problem, the recorded fallback is to monkeypatch the op's retry
policy in that one test; take that only if measurement says it is needed.

## Verification commands

The green-before-push set, all from the worktree root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

`mkdocs build --strict` matters here because the docs changes add cross-page links between
`production-deployment.md`, `sentry.md` and `operations.md`. No network-marked tests are implicated.

Time `uv run pytest` before and after, and report both numbers in the PR — the retry policy adds
wall-clock to three tests by design, and the reviewer should see how much.

## Risks and open questions

1. **A `RetryPolicy` makes a cancelled run take up to ~14 s to stop** (fact 4, verified). *My
   recommendation: accept it.* The alternative is an in-band `RetryRequested` wrapped in the repo's
   existing `except BaseException` / re-raise-interrupts idiom, which keeps cancellation instant —
   but it costs a second guard block in the asset body, only covers the region it wraps, and hands
   the failure hook a `RetryRequested` instead of the real exception (fact 3), which is precisely
   the categorisation defect Change 3 exists to fix. The short delay bounds the cost to seconds.
2. **Is the ~14 s retry budget right?** *My recommendation: yes, for the reason in the assets.py
   section — the hourly back-fill, not the retry, is what protects the data, so the retry only has
   to cover an instantaneous glitch.* If the reviewer wants a budget measured in minutes instead,
   `test_power_time_series_and_metadata_gives_up_after_its_retry_budget` should be dropped rather
   than made to sleep, and replaced by an assertion on
   `power_time_series_and_metadata.op.retry_policy`.
3. **Should blame — upstream vs our code — be tagged at all?** This plan says no, and explains why
   an exception-type allowlist cannot be kept honest. A design that *would* work is a project-defined
   `UpstreamDataError` raised at the boundary where we call the outside world (the NGED S3 read in
   `packages/nged_data/storage.py`, the Dynamical.org read in `packages/dynamical_data/`), with the
   hook walking `__cause__` and tagging `fault_source`. It does not drift, because each boundary
   declares its own status locally. *My recommendation: not now* — it touches two packages, one of
   which (`dynamical_data`) is #506's territory, and its value drops sharply once the retry policy
   removes the common case. Worth its own issue if triage in Sentry still feels slow after this
   ships.
4. **Should `live_forecasts` and `ecmwf_ens` get retry policies too?** Out of scope here.
   `ecmwf_ens` already has its own in-band retry for the condition that warrants one, and
   `live_forecasts` sits in #528's territory. *Recommendation: leave both; open a separate issue if
   their Sentry events prove noisy.*
5. **Should `check_degraded` events drop to `level="warning"`?** It would use Sentry's native
   urgency axis and need no vocabulary. *My recommendation: no* — the existing
   `report_check_degradation` docstring argues these are real faults that deserve an error-level
   event, and `warning` is already taken by the freshness signal, which means something quite
   different. The tag carries the urgency without overloading `level`.
