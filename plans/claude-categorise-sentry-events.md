# Categorise Sentry events (#488)

**The problem.** A transient object-store error during the hourly `power_time_series_and_metadata`
run reaches Sentry looking exactly like a `TypeError` in our own code, even though no data is lost
and the next hourly run back-fills. Separately, when a run *does* fail after exhausting an in-band
retry, Sentry groups it under `RetryRequested` rather than under the exception that actually
happened — so the issue title says "something retried" instead of "the upstream NWP run never
published".

**The planned solution.** Four small changes, none of which needs a list of "upstream" exception
types to keep honest:

1. **A bounded in-band retry around the S3-facing region of `power_time_series_and_metadata`** —
   `raise RetryRequested(...) from exc`, exactly as `ecmwf_ens` already does at `assets.py:370-372`
   — so a transient blip never becomes a Sentry event at all. **Not** a Dagster `RetryPolicy` on the
   asset: that mechanism was tried, probed, and rejected on evidence (see "The mechanism, and why
   the obvious one is wrong" below — it makes a cancelled run finish green).
2. `sentry_capture_failure` reports the underlying cause when Dagster hands it a `RetryRequested`,
   so an exhausted-retry failure groups under the real exception. This is what makes change 1's
   mechanism safe to use, and it fixes a live mis-grouping for `ecmwf_ens` today.
3. `sentry_capture_failure` sends nothing when the exception is a deliberate process exit or
   cancellation, matching the guard idiom the asset and every check already use.
4. One `fault_category: run_failed` tag on the failure hook's events, giving the one class that
   should alert a human a *positive* marker to key an alert rule off, rather than an eliminative
   rule that silently misclassifies the day a fifth sender is added.

Changes 2–4 are confined to `src/nged_substation_forecast/_sentry.py`. Change 1 touches about a
dozen lines of the `power_time_series_and_metadata` body. `defs/checks.py` is untouched and no
public signature changes.

## Verdict, size and departures

**Verdict: worth implementing, and the issue comment's ordering is right.** Its weak preference —
retry first, categorise only what still gets through — is what this plan does. The retry is the
larger win because it removes the commonest false alert rather than labelling it.

**Size: complex.** It changes the production failure behaviour of the hourly ingest job, which is a
degradation rule under `docs/design-philosophy/inherent-stability.md`, and more than one design
defensibly satisfies the issue. It gets this plan, both plan reviews (both run), and both diff
reviews in `implement-issue`.

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
  goes for class 2's `degraded_asset` tag and the freshness warning's `level="warning"` plus
  per-environment fingerprint. Only the failure hook, which carries no distinguishing mark today,
  gains a tag.
- **The plan adds two changes the issue does not mention** (changes 2 and 3). Both are
  categorisation defects of exactly the kind the issue is about, both are verified live behaviour
  rather than theory, and together they are about six lines.

**Departure from this session's wave brief:** the brief says to keep the `defs/assets.py` diff to
the decorator at :75 and the degradation call at :146, on the assumption that the retry would be a
decorator argument. Change 1 instead edits the asset *body*, roughly lines 96–119. Two of the three
territory constraints the brief named have since expired — #580 merged as PR #589 and #528 as
PR #591, both of which this branch has merged in — and the remaining one, #506's NWP region
(~:300–680), is nowhere near the power asset's body. So there is no collision, but it is a departure
from the brief as written, and is flagged as open question 1 for the human to confirm.

## The mechanism, and why the obvious one is wrong

The issue's comment proposes "a short `RetryPolicy` with backoff on the asset". That is the obvious
answer, and probing shows it is unsafe here. **A `RetryPolicy` makes an operator-cancelled run
finish green, having written its data.**

`dagster/_core/execution/plan/utils.py:94-102` catches `(DagsterExecutionInterruptedError,
KeyboardInterrupt)` and re-raises it as a `RetryRequested` when the op carries a policy — commented
"respect retry policy when interrupts occur". Dagster delivers a termination as **one** interrupt
(`_utils/__init__.py:353-360` calls `send_interrupt()` once; `_utils/interrupts.py:80-92` clears the
flag on entry), so that single interrupt is consumed by the conversion, the step restarts, no second
signal ever arrives, and the retried attempt runs to completion. Probed with
`RetryPolicy(max_retries=2, delay=1, backoff=EXPONENTIAL)` and a one-shot interrupt:

```text
ONE-SHOT INTERRUPT + POLICY: run_success=True attempts=2 side_effects=['WROTE DATA']
```

That directly defeats `assets.py:144`'s `raise  # A cancelled run must cancel.` — a line this
repo maintains deliberately, with its own regression test — and it is a far worse fault than the
alert noise it was meant to fix. It is invisible to a probe that raises the interrupt on *every*
attempt, which is why it is easy to miss.

**An in-band `RetryRequested` has none of this**, and is strictly better on three counts:

1. **Cancellation still cancels.** A directly-raised `RetryRequested` short-circuits at
   `plan/utils.py:64-66` before the policy branch is reached, and with no policy on the op an
   interrupt propagates through `plan/utils.py:99-102`'s `else: raise ie`. The existing
   cancellation test keeps both its meaning and its 0.04 s runtime.
2. **Only the upstream-facing region is retried.** A `RetryPolicy` retries the *whole body*, so a
   deterministic bug in our own code *after* the Delta append would be retried, dedupe to a no-op,
   and turn a real failure into a green run — every hour, forever, with nothing in Sentry (probed:
   `run success: True, hook fired: 0`). Wrapping only the S3 read means a bug after that point still
   fails the step and still reports.
3. **It matches the precedent already in this file.** `ecmwf_ens` uses in-band `RetryRequested` for
   "upstream not ready" (`assets.py:366-372`), and reviewers of this file will read one idiom rather
   than two.

The one cost of in-band retry — that the failure hook sees a `RetryRequested` instead of the real
exception — is exactly what change 2 fixes, and change 2 is worth landing on its own account
because `ecmwf_ens` has that defect today. The objection cancels itself.

## Verified facts this design rests on

Established by probing the installed Dagster 1.13.17 and reading its source. Each was checked twice,
independently.

1. **A `failure_hook` does not fire on an attempt that will be retried.** A step going up for retry
   emits `STEP_UP_FOR_RETRY`, and `failure_hook` tests `event.is_step_failure`, which is
   `STEP_FAILURE` only (`hook_decorator.py:259`, `events/__init__.py:660`). Probed: 3 attempts, hook
   fired **0** times, run succeeded. This is what makes change 1 *remove* the event.
2. **After the retries are exhausted the hook fires exactly once.** For a `RetryPolicy` it sees the
   original exception, because `HookContext.op_exception` already unwraps `RetryRequestedFromPolicy`
   to its `__cause__` (`_core/execution/context/hook.py:130-137`).
3. **That upstream unwrap covers only `RetryRequestedFromPolicy`, not a plain `RetryRequested`**
   (`plan/utils.py:30` defines the subclass; `:64-66` short-circuits the plain one). So a
   hand-raised `RetryRequested` reaches the hook as itself. This is live today for `ecmwf_ens`:
   a run that never publishes exhausts its 8 retries and reaches Sentry **titled and grouped** as
   `RetryRequested`. The `__cause__` chain *is* serialised into the event, so the detail is not
   lost — what is lost is the title and the grouping, which is what an operator triages on.
   Change 2 extends Dagster's own unwrap one class up the hierarchy; note it also drops the
   `RetryRequested` frame from the reported exception, so "this was a retry exhaustion" has to be
   read from the retry count in Dagster rather than from the Sentry title.
4. **A `RetryPolicy` converts an interrupt into a bare `RetryRequested`** (`plan/utils.py:94-102`),
   and Dagster delivers a termination as one interrupt only — hence the green-cancelled-run defect
   above. `DagsterExecutionInterruptedError` is not a `DagsterError`, so it falls past the
   `except DagsterError` clause at `plan/utils.py:60`.
5. **Today, with no policy, cancellation does not reach the failure hook — but `SystemExit`
   does.** Probed with no policy: `DagsterExecutionInterruptedError` → 0 hook fires,
   `KeyboardInterrupt` → 0 (both re-raise at `execute_plan.py:319-326` before `_trigger_hook` at
   `:99` is reached), **`SystemExit` → 1 hook fire** (it is neither a `DagsterError`, nor an
   `Exception`, nor in the interrupt clause, so it reaches `execute_plan.py:328`'s
   `except BaseException` and yields a `STEP_FAILURE` without re-raising). Change 3 is justified by
   this verified case, not by the retry mechanism.
6. **`build_hook_context` refuses a `BaseException`.** `context/hook.py:484` runs
   `check.opt_inst_param(op_exception, "op_exception", Exception)`. Probed:
   `DagsterExecutionInterruptedError`, `KeyboardInterrupt` and `SystemExit` are all **rejected** at
   construction; `RetryRequested` is accepted. The tests for change 3 therefore need a duck-typed
   context stub — the hook touches only `.op_exception` — and the plan budgets for it below.
7. **`RetryRequested` takes a flat `seconds_to_wait`**; there is no backoff for the in-band form.
   (For the record, `RetryPolicy`'s exponential backoff is `delay × (2ⁿ − 1)`, not `delay × 2ⁿ⁻¹` —
   `policy.py:99-100` — which is worth knowing because it is an easy factor-of-two error.)
8. **Retries are enabled under `materialize()` / `execute_in_process`**, so the existing tests in
   `tests/test_assets.py` exercise a retry rather than silently skipping it.

## What changes, file by file

### `src/nged_substation_forecast/defs/assets.py` — the in-band retry

Two `Final` constants beside the existing `_ECMWF_ENS_*` pair, named and documented to match:
`_POWER_INGEST_MAX_RETRIES = 2` and `_POWER_INGEST_RETRY_DELAY_SECONDS = 2`.

The S3-facing region — `list_timeseries_json_files`, `remove_small_files_from_listing`,
`select_new_rows` and `download_and_parse_files` (`assets.py:96-119`) — is wrapped so that anything
it raises becomes a bounded retry, using the guard idiom this repo already uses at `assets.py:140-144`
and `checks.py:426-432`:

- `except NoNewData` keeps its existing early-return behaviour and **must be handled before the
  retry guard**, or a normal "nothing new on S3" hour would start retrying.
- `except BaseException as exc`: re-raise `KeyboardInterrupt | SystemExit |
  DagsterExecutionInterruptedError` unchanged, so a cancelled run still cancels; otherwise
  `context.log.warning(...)` and `raise RetryRequested(max_retries=..., seconds_to_wait=...) from exc`.

`BaseException` rather than a list of upstream exception types, for the reason `checks.py:426-431`
already gives: obstore, delta-rs and polars each define their own classes and a pyo3 panic is not an
`Exception`, so naming what must *propagate* is the only version that stays true as dependencies
come and go. `RetryRequested` needs no new import — it is already imported at `assets.py:36` for
`ecmwf_ens`.

**The budget is 2 retries × 2 s = 4 s of waiting, and is deliberately short.** The whole value of
the retry is suppressing an alert; the data is never at risk, because the asset is unpartitioned,
re-lists NGED's S3 bucket from scratch each attempt, and the next hourly run back-fills whatever
this one missed. A longer budget buys nothing the next hour does not already buy. A persistent
outage still fails after 4 s and alerts, which is the wanted behaviour. This is rule 10 of
`inherent-stability.md` — bounded retries with backoff — at its cheapest useful setting.

**The real cost of a retry is not the 4 s of sleep** but the two extra complete ingest passes it
implies: `RetryRequested` restarts the whole step, so the S3 listing, download and parse all re-run.
At V2 scale (~2,500 series) that is the dominant term. The docs paragraph should say so, and it is
another reason to keep the retry count at 2.

**Retrying is safe, but not for the reason it first appears.** The Delta append at :155-160 is *not*
the last statement in the body — the summary metadata at :163-171 follows it. Idempotency comes from
`select_new_rows` instead, which filters on `time > last_time`, so a second attempt appends nothing
even though the file listing still offers the same file; and `upsert_metadata` is an upsert. In any
case, under change 1 a failure after the append cannot trigger a retry at all, because it is outside
the guarded region.

**Nothing else in `assets.py` is touched** — not the `report_asset_degradation` call at :146, and
nothing in the NWP region.

### `src/nged_substation_forecast/_sentry.py` — three changes, all in `sentry_capture_failure`

The function currently captures `context.op_exception` whenever it is not `None` (:119-132). It
gains, in order:

1. **The `RetryRequested` unwrap.** When the exception is a `dagster.RetryRequested` with a
   non-`None` `__cause__`, work with the cause instead. The comment should cite
   `HookContext.op_exception`'s own unwrap at `hook.py:134` and say this extends it to the plain
   class, which `assets.py:370` — and now the power ingest — raises. Must tolerate `__cause__ is
   None` without crashing. `isinstance` against `RetryRequested` covers `RetryRequestedFromPolicy`
   too, since it is a subclass.
2. **The deliberate-exit guard.** If the resulting exception is a `KeyboardInterrupt | SystemExit |
   DagsterExecutionInterruptedError`, return without capturing. The justification in the comment must
   be fact 5 — that `SystemExit` reaches this hook today, verified — and the repo's existing idiom at
   `assets.py:144` / `checks.py:432`, which already treats all three as "must cancel, not a fault".
   Do **not** write that the retry mechanism routes cancellations here: with the in-band form it
   does not, and a comment that says so would be false and would invite deletion.
3. **The `fault_category: run_failed` tag**, with the value as a `Final` module constant. **Set it by
   calling the existing `_capture_tagged` (:135-151)** rather than opening a fresh
   `sentry_sdk.new_scope()`: that function already forks the scope, sets one tag, captures, and logs
   at ERROR if Sentry itself fails — and it is already covered by
   `test_degradation_reporters_capture_the_exception_and_tag_the_name` (`test_sentry.py:193`).
   Re-implementing it inline would leave the hook the only sender that does not log when telemetry
   breaks.

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

Four passages. **`docs/design-philosophy/inherent-stability.md` is not edited** — this change
applies its existing rules rather than adding one.

- **`docs/architecture/production-deployment.md`**, § "Send telemetry to Sentry, and alarm on
  absence" (~lines 194–260): one paragraph on the ingest retry (a blip is not an event; the budget
  is short because the next hourly run back-fills; the real cost is two extra ingest passes, not the
  4 s), one sentence that a deliberate exit is not reported, and one recording that we do *not*
  classify upstream-vs-ours by exception type, with the drift argument.
- **`docs/live_service/sentry.md`**: a fourth console step under "Turn it on in production" giving
  the alert-rule recipe — alert on `fault_category:run_failed`; route `asset_check:*`,
  `degraded_asset:*` and `level:warning` to a non-alerting channel. Phrase urgency in the vocabulary
  `inherent-stability.md` already uses ("next business day"), **not** as a page: that page's
  failure-mode table uses only "No" and "Yes, next business day", and line 153 states "Nothing here
  is a 2am page."
- **`docs/live_service/operations.md`**: beside the existing `asset_check:…` / `degraded_asset:…`
  runbook mentions (~lines 227, 234, 306), one clause noting that a transient object-store error on
  the hourly ingest is retried twice before it reports.
- **`docs/live_service/aws.md`** (~lines 311–315, "that schedule fails every hour — loudly"): still
  true, but it now takes three attempts to get there. One clause.

## Design-philosophy check

This path is **production**, so it must degrade rather than raise, and it does:

- The retry strictly *reduces* production alerts on the upstream-read path and cannot introduce a
  failure. A persistent fault still fails after 4 s exactly as today.
- **A cancelled run still cancels** — the property the `RetryPolicy` mechanism would have broken,
  and the reason it was rejected. `assets.py:144` and its regression test keep their meaning.
- **No warning path gains the ability to raise.** `_capture_tagged` and `report_power_freshness` are
  unchanged, so their existing guards hold. `sentry_capture_failure` now routes through
  `_capture_tagged`, which is itself guarded; and it runs inside Dagster's hook error boundary (a
  raising hook yields `HOOK_ERRORED`, not a run failure — `execute_plan.py:127-152`), on a step that
  has already failed. It stays a no-op with an empty DSN because `capture_exception` needs an active
  client.
- **Liberal about missing inputs, strict about malformed ones**: a transient object-store error is
  the outside world misbehaving, so retrying and carrying on is the required response.
- No asset check is added or changed, so the `WARN`/`blocking=False` rule is not engaged.
- Rule 10 (bounded retries with backoff) is the rule this implements. Cite the exact
  `engineering-hypotheses.md` label when the docs edit lands rather than guessing at it here — the
  labels are append-only and must not be misapplied.

Against `design-principles.md`, nothing is traded away. The debit is that a *bug inside the guarded
S3 region* is now retried twice before it reports, costing 4 s and two extra ingest passes. Bounded,
and it still reports.

## Tests

**`tests/test_sentry.py`.** The existing stubbing machinery in
`test_degradation_reporters_capture_the_exception_and_tag_the_name` (:193) and
`test_report_power_freshness_does_not_leak_scope` (:388) covers most of this, but per fact 6 the
cancellation tests need a **duck-typed context stub** (a small object exposing only `.op_exception`)
because `build_hook_context` rejects a `BaseException`. Budget for that stub.

1. `test_failure_hook_reports_the_cause_of_a_retry_requested` — a `RetryRequested` whose `__cause__`
   is an `OSError`; assert the captured exception **is** the `OSError`. *Fails on `main`*: today the
   `RetryRequested` itself is captured.
2. `test_failure_hook_captures_a_retry_requested_with_no_cause` — the same class with `__cause__`
   unset is captured as itself. **This one passes on `main`**; it is a guard against a `None`
   dereference in the new code, not a test of a behaviour change, and is kept for that reason.
3. `test_failure_hook_ignores_a_deliberate_exit` — parametrised over `SystemExit` (the shape fact 5
   shows reaches the hook today), `DagsterExecutionInterruptedError`, `KeyboardInterrupt`, and a
   `RetryRequested` wrapping one; assert `capture_exception` is never called. *Fails on `main`* for
   `SystemExit` and for the wrapped case; the other two are defence in depth.
4. `test_failure_hook_tags_the_fault_category` — assert the tag value and that it is set on a forked
   scope. *Fails on `main`*: the hook sets no tag and forks no scope.

**`tests/test_assets.py`**, in the `test_power_time_series_and_metadata_*` block (:192–371 — this
session's territory; #580 owns only the `test_ecmwf_ens_*` block):

1. `test_power_time_series_and_metadata_retries_a_transient_upstream_failure` — stub
   `download_and_parse_files` (inside the guarded region, and the call that actually crosses the
   network for content) to raise `OSError` on its first call and delegate to the real function
   afterwards; assert the run succeeded, the stub was called twice, and the power table and metadata
   parquet match what the happy-path test asserts. *Fails on `main`*: without the retry the first
   `OSError` fails the run. Costs ~2 s (one retry delay).
2. `test_power_time_series_and_metadata_gives_up_after_its_retry_budget` — stub the same function to
   raise every time; assert the run failed and the stub was called exactly
   `_POWER_INGEST_MAX_RETRIES + 1` times. *Fails on `main`*: 1 call, not 3. This is what pins the
   budget down; unlike the `RetryPolicy` version it is now testing *our* `raise RetryRequested`, not
   Dagster's machinery. Costs ~4 s.
3. `test_power_time_series_and_metadata_does_not_retry_no_new_data` — with `download_and_parse_files`
   raising `NoNewData`, assert it is called **once** and the run succeeds. *Fails on `main`* only
   against a wrong implementation that puts the retry guard before the `NoNewData` handler — which
   is the specific ordering mistake this change can make, so it is worth its ~0 s.
4. `test_power_time_series_and_metadata_does_not_retry_a_cancelled_run` — stub
   `download_and_parse_files` to raise `DagsterExecutionInterruptedError`; assert it is called once
   and the run fails. *Fails on `main`* only against a wrong implementation, and it is the direct
   regression test for the defect that sank the `RetryPolicy` design. Cheap and worth having.

**Existing test that must keep its meaning**:
`test_power_time_series_and_metadata_re_raises_a_cancelled_run` (:280) stubs `upsert_metadata`,
which is *outside* the guarded region, so it is unaffected in both behaviour and runtime. Confirm
that when implementing — under the rejected `RetryPolicy` design it would have stayed green while
the property it documents became false in production, which is precisely the trap this plan exists
to avoid.

**A property deliberately not tested**: idempotency after a partial write. Under change 1 a failure
after the Delta append cannot request a retry, so the case is unreachable by construction rather
than guarded by a test.

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
`production-deployment.md`, `sentry.md`, `operations.md` and `aws.md`. No network-marked tests are
implicated. Baseline for comparison: the suite is 225 passed in ~40 s on this branch today; the new
tests add ~6 s. Report both numbers in the PR.

## Risks and open questions

1. **Change 1 edits the `power_time_series_and_metadata` body, which the wave brief asked me to keep
   to the decorator at :75.** The brief assumed a `RetryPolicy`, which the evidence above rules out.
   The edited region (~:95–120) collides with neither #580 nor #506. *Recommendation: proceed; this
   is a mechanism change the brief could not have anticipated, not a scope expansion.* Worth an
   explicit yes before implementation starts.
2. **A bug inside the guarded S3 region is now retried twice before it reports.** *Recommendation:
   accept.* It is bounded at 4 s plus two ingest passes, it still reports, and the alternative —
   naming the upstream exception types — is the drifting list the issue itself warns about.
3. **Should blame — upstream vs our code — be tagged at all?** This plan says no, and says why an
   exception-type allowlist cannot be kept honest. A design that *would* work is a project-defined
   `UpstreamDataError` raised at each boundary where we call the outside world (the NGED S3 read in
   `packages/nged_data/storage.py`, the Dynamical.org read in `packages/dynamical_data/`), with the
   hook walking `__cause__` and tagging `fault_source` — it cannot drift, because each boundary
   declares its own status locally. *Recommendation: not now.* It touches two packages, one of which
   is #506's territory, and its value drops sharply once change 1 removes the common case. Worth its
   own issue if Sentry triage still feels slow after this ships.
4. **Should `live_forecasts` get the same treatment?** Out of scope — it is #528's territory.
   *Recommendation: leave it; open a separate issue if its events prove noisy.*
5. **`RetryPolicy` is now demonstrably unsafe on any asset with a meaningful cancellation contract.**
   That is a general finding, not specific to this issue, and no asset in the repo uses one today.
   *Recommendation: capture it as a short note in the docs paragraph so the next person reaching for
   a `RetryPolicy` finds the evidence, rather than opening a separate issue about a thing we do not
   do.*

## What the two plan reviews changed, and what was rejected

**Review 1 (simplicity)** cut the design by about three quarters and caught a factual error.
Accepted: dropping a four-value `fault_category` vocabulary down to one tag on the one sender that
lacks a mark (the other three are already distinguishable by `degraded_asset`, `asset_check` and
`level`+fingerprint); the backoff arithmetic error; citing `hook.py:134`; trimming the docs surface;
leaving `inherent-stability.md` alone; and dropping an urgency ladder ("Page." / "same working day")
that contradicted that page's own two-value column and its line 153, "Nothing here is a 2am page."

Rejected from review 1, with reasons:

- **"Cut the tag entirely; identify the alerting class by elimination."** Rejected for the one sender
  that alerts a human, accepted for the other three. An eliminative rule is correct today and fails
  silently the day a fifth sender is added, and it is the alerting rule — the one place that
  fragility is least affordable. Cost of the positive marker: about five lines.
- **"Use Sentry's native `level` instead of a tag."** Rejected: `level:warning` already means
  "upstream telemetry is late" on the freshness event, so reusing it for check degradation would
  merge two unrelated meanings onto one facet.
- **"A cancelled run sends a Sentry event today, so the interrupt guard fixes a pre-existing noise
  class."** Rejected as stated — probing showed 0 hook fires for `DagsterExecutionInterruptedError`
  and `KeyboardInterrupt`. Review 2 then showed the finding was *partly* right for a shape neither
  review had isolated: `SystemExit` does reach the hook. Change 3 now rests on that verified case.

**Review 2 (correctness)** overturned the plan's central mechanism. Accepted: the green-cancelled-run
defect that sank `RetryPolicy` (F1); that `build_hook_context` refuses a `BaseException`, so the
cancellation tests need a duck-typed stub (F2); that a `RetryPolicy` would silently heal a real bug
occurring after the Delta append (F3); the `SystemExit` case (F6); that "the Delta append is the last
statement" was false and the real idempotency mechanism is `select_new_rows`' `time > last_time`
filter (F7); reusing `_capture_tagged` instead of a bare `new_scope()` (F8); that change 2 fixes the
title and grouping rather than recovering lost detail, and drops the `RetryRequested` frame (F9);
that the budget's real cost is two extra ingest passes (F11); dropping the false blanket claim that
every proposed test fails on `main` (F5); and the missing `aws.md` doc clause.

Rejected from review 2:

- **"Rewrite `test_power_time_series_and_metadata_re_raises_a_cancelled_run` to assert the new,
  worse behaviour" (F4, option b).** Rejected because the plan takes the reviewer's option (a)
  instead — switching mechanism — under which that test keeps its original meaning and needs no
  change. Writing a test to document a regression would have been the wrong way out.
- **"The retry test only exercises trivial idempotency" (F7, second half).** Accepted in substance
  and resolved differently: the stub moves from `list_timeseries_json_files` to
  `download_and_parse_files`, and the non-trivial case is documented as unreachable by construction
  rather than tested, because under in-band retry it genuinely is.
