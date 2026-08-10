# Issue #480 — make `power_data_is_fresh` degrade instead of raising

Mechanical checklist for the in-flight PR. Delete on merge (paste into the PR body first).

## Verdict: worth implementing; the issue body is accurate, its framing needs one correction

Every premise was re-derived against the code and against Dagster 1.13.17, then re-attacked by an
independent reviewer. What holds:

1. **The check is its own step inside the hooked job.** Resolving the real definitions,
   `power_time_series_and_metadata_job` has steps
   `['power_time_series_and_metadata_power_data_is_fresh', 'power_time_series_and_metadata']` and
   carries `hooks={sentry_capture_failure}`.
2. **A raise inside a `blocking=False` `@asset_check` fails the run *and* fires the failure
   hook.** A probe job with one raising non-blocking check reported `run success: False` and
   `hooks fired: ["my_asset_raising_check: RuntimeError('boom')"]`, while a merely-*failing*
   check did not. `blocking=False` governs whether a failed check blocks downstream assets; it
   does nothing about an erroring check step.
3. **A corrupt `metadata.parquet` propagates out of the check today**, as
   `polars.exceptions.ComputeError: parquet: File out of specification: The file must end with
   PAR1`, from `_read_roster_ids` via `checks.py:279`.

**The correction.** The issue leaves the impression that a corrupt `metadata.parquet` is *the*
motivating fault and that this fix cures it. Neither is quite right, and the PR body must not
repeat the claim:

- Dagster **skips a check whose asset op failed** (verified: asset raises → `check evaluations:
  []`). And `upsert_metadata` (`packages/nged_data/src/nged_data/storage.py:404-406`) does its own
  unguarded `pl.read_parquet(metadata_path, …)`. So in any hour where new data *arrives*, the
  asset dies on the corrupt roster first and the check never runs. This fix does not make that
  hour green; guarding the asset-side read is a separate question (follow-up below).
- The hours this fix covers are the ones where `download_and_parse_files` raises `NoNewData` and
  the asset returns early (`defs/assets.py:105-111`) — which is *most* hours, since NGED publishes
  roughly every 6 hours and this job runs hourly. In those hours the check is the sole raiser.
- The likelier raiser in practice is not corrupt parquet at all: it is a transient object-store
  error inside `time_series_coverage`'s Delta scan, or a future bug anywhere in the check. Those
  need no corrupted file and no unusual state, and today each one fails the hourly production run
  and pages via Sentry.

That is a genuine violation of
[inherent-stability rule 7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
in the one job that runs hourly, and the fix is about fifteen lines. Worth doing.

## Scope

`power_data_is_fresh`, plus one consistency change to `live_forecasts_are_healthy` (step 1c —
Jack approved it). The pure `evaluate_power_freshness` is untouched, so this does not collide with
[#420](https://github.com/openclimatefix/nged-substation-forecast/issues/420).

## 1. `src/nged_substation_forecast/defs/checks.py`

**a. Extract the check body** into a private helper directly above the check (mirrors
`_evaluate_live_forecasts` at `checks.py:802`):

```python
def _check_power_data_freshness() -> AssetCheckResult:
    """Read the power table's recency off disk and judge it.

    Split out from the check itself so the check's ``except`` wraps everything — the ``Settings``
    load, both reads, the evaluation, the Sentry report and the metadata build — rather than only
    part of it.
    """
    settings = Settings()
    storage_options = settings.storage_options
    coverage = time_series_coverage(settings.power_time_series_data_path, storage_options)
    roster_ids = _read_roster_ids(settings.metadata_path, storage_options)
    result = evaluate_power_freshness(
        coverage=coverage,
        roster_ids=roster_ids,
        now=datetime.now(UTC),
        threshold=_POWER_DATA_STALENESS_THRESHOLD,
    )
    # Forward per-series staleness to Sentry (a no-op unless a DSN is set and some series is late).
    # Best-effort: report_power_freshness never raises, so a telemetry hiccup can't fail this check.
    report_power_freshness(settings, result)
    return _to_asset_check_result(result)
```

Named `_check_power_data_freshness`, deliberately *not* `_evaluate_power_data_freshness`: the
module already exports the pure `evaluate_power_freshness` and two names differing by one word
would be a trap.

**b. `power_data_is_fresh` (`checks.py:270`) becomes the catch-all wrapper**, following
`live_forecasts_are_healthy` (`checks.py:847-860`):

```python
def power_data_is_fresh() -> AssetCheckResult:
    """Report how many time series are late on the ``power_time_series`` Delta table.

    Runs automatically alongside every ``power_time_series_and_metadata`` materialisation (hourly
    via ``power_time_series_and_metadata_schedule``), so the check re-evaluates freshness each
    hour regardless of whether new data landed.

    Cannot fail its own step: the whole body is guarded, so a stalled object store, a half-written
    ``metadata.parquet`` or a bug in here degrades to an unhealthy result instead of failing the
    hourly production run.
    """
    try:
        return _check_power_data_freshness()
    except (Exception, PanicException) as exc:
        # Catch-all is deliberate. This check is non-blocking, but it runs as its own step inside
        # `power_time_series_and_metadata_job`, whose `sentry_capture_failure` hook would turn a
        # raise here into a failed production run — fail-open silently becoming fail-closed.
        # `PanicException` is named explicitly because pyo3 derives it from `BaseException`, not
        # `Exception`, so a Rust panic inside a Polars or delta-rs read would otherwise sail
        # straight past this handler. Dagster's own interrupt errors are `BaseException` too and
        # deliberately keep propagating: a cancelled run should cancel. Logged at ERROR with the
        # traceback (never a silent swallow) and surfaced as an unhealthy check.
        logger.exception("Could not evaluate power-data freshness")
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=f"Could not evaluate power-data freshness: {exc!r}",
        )
```

`from polars.exceptions import PanicException` at the top. Verified: `PanicException.__mro__` is
`(pyo3_runtime.PanicException, BaseException, object)`, and `ruff check --select BLE,TRY,ANN` plus
`ty check` are both clean on this construct.

**c. Same two-line change to `live_forecasts_are_healthy` (`checks.py:849`)** — add
`PanicException` to its handler. It reads Delta through the same pyo3 stack, the docs already
claim it "cannot raise", and leaving one of the two half-guarded while the module docstring
asserts a shared property would be worse than the status quo. This is the one piece of scope
beyond the issue; say the word and I will drop it and weaken the doc wording instead.

**d. Module docstring, final paragraph (`checks.py:33-39`)**, which currently singles
`live_forecasts_are_healthy` out. Replace with a statement that is *true of both* and does not
over-claim:

> Both checks are `AssetCheckSeverity.WARN` and `blocking=False`, and neither can fail its own
> step: each one's whole body sits under a catch-all for `Exception` and for pyo3's
> `PanicException`, which logs the traceback and returns an unhealthy result. Only Dagster's own
> interrupt errors propagate, which is what we want — a cancelled run should cancel. A warning
> path that could fail would turn fail-open into fail-closed at exactly the wrong moment (rule 7
> of [The rules](…)).

Drop the "every read it makes is guarded" clause from the shared sentence: that is
`live_forecasts_are_healthy`-only behaviour, which `power_data_is_fresh` deliberately does *not*
adopt (rejected alternative 1). Keep it as a sentence about that check alone if it earns its place.

## 2. Tests — `tests/test_checks.py`

Three new tests. All three were written into a scratchpad and **run against the current unfixed
code**: (a) and (b) error with the exception propagating, (c) fails on `assert result.success`
with `DagsterExecutionStepExecutionError` on the check step. Put (a) and (b) after
`test_power_data_is_fresh_hands_evaluated_result_to_sentry` (~line 284).

**(a) A bug inside the check surfaces as a warning.** Mirrors
`test_live_forecasts_check_contains_an_internal_error` (`test_checks.py:967`):

```python
def test_power_data_is_fresh_contains_an_internal_error(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a bug inside the check itself must surface as a warning, never as a raise."""

    def _boom(**_kwargs: object) -> PowerFreshnessResult:
        raise RuntimeError("simulated bug inside the check")

    monkeypatch.setattr(checks, "evaluate_power_freshness", _boom)
    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert "simulated bug inside the check" in str(result.description)
```

No fixture data needed: with no tables at all, `time_series_coverage` returns an empty frame and
`_read_roster_ids` returns `None`, so the patched evaluator is still reached. The `**_kwargs`
signature matches the call site (four keyword arguments), and the module-global lookup survives
the extraction into `_check_power_data_freshness`.

**(b) An unreadable roster.**

```python
def test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet(env: Path) -> None:
    """A half-written roster is a realistic on-disk raiser: `metadata.parquet` is written in place,
    so a process killed mid-write leaves a file that exists and will not parse. This check must
    warn rather than raise — it is one step of the hooked `power_time_series_and_metadata_job`.
    (It is not the *only* raiser on that state: in an hour where new data arrives, `upsert_metadata`
    hits the same file first and fails the asset. This test pins the check's half.)"""
    settings = Settings()
    # …write a one-row power_time_series Delta table exactly as the tests above do…
    Path(settings.metadata_path).write_bytes(b"not a parquet file")

    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert "Could not evaluate power-data freshness" in str(result.description)
```

Assert on our own description prefix, not on `ComputeError` or `"PAR1"` — the Polars message is
not ours to pin.

**(c) The property in the issue's title: the run does not fail.** (a) and (b) pin the *return
value*; only this pins that a Dagster run containing the check still succeeds. Needs no S3 and no
asset execution — the check runs as its own step in ~13 ms:

```python
def test_power_data_is_fresh_never_fails_the_run(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The whole point of the catch-all: a raise inside this check fails the *step*, and so the
    hourly run, and so pages via `power_time_series_and_metadata_job`'s Sentry failure hook — even
    though the check is `blocking=False`, which governs only whether a *failed* check blocks
    downstream assets."""

    def _boom(**_kwargs: object) -> PowerFreshnessResult:
        raise RuntimeError("simulated bug inside the check")

    monkeypatch.setattr(checks, "evaluate_power_freshness", _boom)
    result = materialize(
        [checks.power_data_is_fresh],
        selection=AssetSelection.checks(checks.power_data_is_fresh),
        instance=dagster_instance,
        raise_on_error=False,
    )
    assert result.success
    (evaluation,) = result.get_asset_check_evaluations()
    assert evaluation.passed is False
    assert evaluation.severity == AssetCheckSeverity.WARN
```

Verified as written: the target asset does **not** need to be in the list, so no
`power_time_series_and_metadata` import is added. New imports for `test_checks.py`:
`AssetSelection`, `DagsterInstance` and `materialize` from `dagster`. The `dagster_instance`
fixture already exists in `tests/conftest.py`; using it (rather than `DagsterInstance.ephemeral()`
inline) is what keeps the SQLite teardown noise out of the suite.

**(d) `tests/test_checks.py` module docstring** — it says the end-to-end tests drive the real
`@asset_check` directly. Add that one case runs it through Dagster's executor instead, because
asserting run success is the only way to pin the fail-open property.

## 3. Docs

Present tense, no changelog notes.

- **`docs/architecture/production-deployment.md`**
  - `power_data_is_fresh` section (~line 67-85): add that the check cannot fail its own step and
    why, next to the existing "the severity is a warning rather than a failure" reasoning.
  - `live_forecasts_are_healthy` section (~line 124-131) currently reads "Two design points follow
    the `power_data_is_fresh` pattern deliberately… And it **cannot raise**: **every read is
    guarded**, and the whole body sits under a catch-all". Promote *only* "the whole body sits
    under a catch-all" into the shared list; per-read guarding stays a `live_forecasts_are_healthy`
    detail, because `power_data_is_fresh` deliberately does not do it. (Writing "three design
    points followed deliberately" without that split would make the page assert something false.)
- **`docs/design-philosophy/inherent-stability.md`, rule 7 (~lines 182-184)**: "…which is why
  `report_power_freshness` never raises" → name the two asset checks as well. After this change
  the rule is enforced in three places, and this text is what the next check author reads.
- **`docs/live_service/operations.md`, "Reading the freshness check" (~line 161)**: one sentence
  telling the operator that a `Could not evaluate power-data freshness: …` description means the
  check could not read its inputs (suspect the object store, or `metadata.parquet`) — a different
  remedy from the late-series case the paragraph already covers.
- **`src/nged_substation_forecast/_sentry.py:191-200`** (docstring, not code):
  `report_power_freshness`'s rationale currently ends "…would otherwise trip the failure hook and
  fail the run". After this change the check's own catch-all absorbs it, so the *consequence* is
  different, not absent: a raise here would cost the entire freshness report — every late series,
  in the very hour they went late — because the check would degrade to "could not evaluate".
  Rewrite to that. Its `except Exception` stays as it is: unlike the checks, its body builds a
  payload from an already-materialised frame rather than opening a new pyo3 read.

Not changing: `docs/architecture/why-dagster-not-airflow.md` (lines 202, 337) and
`docs/live_service/sentry.md:144-150` — checked, and both stay true. The sentry.md sentence is
conditioned on the check "finding any series past the staleness threshold", which a degraded check
does not.

## Design decisions, and alternatives rejected

1. **Rejected: guard each read individually, the way `live_forecasts_are_healthy` guards
   `meta.json`.** `evaluate_power_freshness` accepts `roster_ids=None` as "never-reported ids are
   undetectable", so degrading a *corrupt* `metadata.parquet` to `None` leaves a fresh
   `power_time_series` table reporting `passed=True` — verified: `n_stale=0, n_never=0,
   n_series_total>0`. A green tick over a corrupt roster is worse than the blanket catch-all's
   `passed=False` naming the exception. The sibling's case differs: its `meta.json` is populated
   out-of-band, its *absence* is normal, and its degradation is visible through omitted metadata
   keys.
2. **Rejected: a tri-state roster read** — distinguish "absent" from "unreadable", still evaluate
   and report staleness, but force `passed=False` with the reason in the description. This is
   strictly more informative than option 1 and answers the real cost of the blanket catch-all,
   which is that an unreadable roster discards the staleness signal entirely (no `n_stale`, no
   late-series table, and `report_power_freshness` never called, so Sentry hears nothing even if
   30 series are genuinely stale). Rejected on cost/benefit rather than on principle: the state it
   improves on — a roster file present but unreadable — already fails the *asset* loudly in every
   hour where data arrives, so the marginal information is small against a third result state to
   carry, describe and test. Worth revisiting if the asset-side read is ever guarded too.
3. **Rejected: factor the log-and-degrade tail into a shared helper.** Ruff's `BLE` family is
   selected (`pyproject.toml:126`) with no relevant ignore, and `BLE001` tolerates
   `except Exception` only when the handler body itself calls `logger.exception` or re-raises.
   Verified with `ruff check --isolated --select BLE` on a two-function probe: inline is clean,
   delegating reports `BLE001`. DRY here buys four lines at the cost of a `# noqa` on the rule
   that exists to catch this construct.
4. **Rejected: a `@_cannot_raise("…")` decorator wrapping each check body.** This *would* satisfy
   BLE001 (the `try`/`except` and its inline `logger.exception` live together inside the
   decorator) and would scale to a third check, so option 3's argument does not dispose of it. It
   is rejected on two other grounds: `asset_check` introspects the decorated function's signature
   to decide whether to pass `context`, and `inspect.getfullargspec` does not follow
   `functools.wraps`' `__wrapped__` — so the wrapper risks changing how Dagster binds `context`,
   for two call sites; and the two checks want different degraded descriptions and different
   rationale in their docstrings, which is most of what the decorator would be hiding. Revisit
   when there is a third check.
5. **Rejected: catching a narrower exception tuple** (`OSError, ComputeError, …`). Rule 7 is about
   *any* failure of the warning path, and the raiser we cannot enumerate is a future bug in
   `_to_asset_check_result` or the Sentry payload build.
6. **Rejected: `except BaseException`.** It would swallow `DagsterExecutionInterruptedError` and
   `KeyboardInterrupt`; a cancelled run must cancel. `PanicException` is named explicitly instead.
7. **Rejected: attaching metadata to the degraded result.** `live_forecasts_are_healthy` omits it
   so each numeric metadata key keeps one type across runs and stays plottable in Dagster; the
   description carries the explanation instead.
8. **Rejected: moving the `report_power_freshness` call outside the guarded region.** It already
   cannot raise, so there is nothing to gain, and keeping it inside means the `except` genuinely
   wraps the whole body, as the helper's docstring claims.

## Known limits of this fix — stated, not silently left

Worth a sentence in the PR body so nobody reads the fix as bigger than it is:

- **The check can still hang rather than raise.** `time_series_coverage` scans the whole Delta
  table with no timeout; a stalled object store blocks the step indefinitely, which is arguably
  worse than failing. Neither this check nor the sibling addresses it, and neither does this PR.
- **The returned value is serialised outside the guard.** Dagster builds the check-evaluation
  event after the function returns, so `_late_table_metadata` (`checks.py:213-227`) is not
  protected. It is also **unbounded**, where the sibling deliberately caps its listing at
  `_MAX_MISSING_SERIES_LISTED = 20` — a total feed outage at V2 scale would emit 2,500
  `TableRecord`s every hour. Not a demonstrated raiser, and out of scope here, but it is a real
  asymmetry with the check this one is being made to match.
- **Resource init for the check step** (`io_manager`) also happens outside the guard. Unchanged by
  this fix.

## Out of scope — flagged for follow-up issues, not fixed

1. **`upsert_metadata`'s unguarded `pl.read_parquet`** (`packages/nged_data/src/nged_data/storage.py:404-406`).
   The corrupt-roster state fails the *asset* in every hour where new data arrives. That is the
   other half of the fault this issue describes, and it needs its own decision (rebuild the roster
   from the new metadata? quarantine the file?) rather than a `try`.
2. **`nwp_has_no_unexpected_nulls` / `nwp_run_is_complete`** (`defs/assets.py`) are computed inside
   the `ecmwf_ens` asset, *after* `write_nwp` has already appended the run. A raise in
   `assess_nwp_quality` or `assess_nwp_run_completeness` therefore fails the asset with the data
   already on disk, and the retry re-appends it. Same rule-7 family, different remedy — it needs
   idempotence thinking, not a `try`.
3. **`_late_table_metadata` is uncapped** (see limits above).

All three are filed under epic #138, positioned immediately after #480: **#508** (the roster read),
**#509** (the NWP checks), **#510** (the uncapped table).

## Verification before pushing

```bash
uv run ruff check . && uv run ruff format . && uv run --all-packages ty check && uv run pytest
```

Docs are touched, so also:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict
```

…then read the rendered HTML under `site/` for the three pages edited — the linters pass on
rendering that is visibly wrong.

Finally, `git stash` the `checks.py` change and confirm all three new tests fail against the
unfixed code (already done once in a scratchpad; repeat on the final wording).

## PR

Branch `claude/github-issue-480-plan-5db0a2` → `main`. Commit messages end with
`Co-Authored-By: Claude <noreply@anthropic.com>`. PR body: Claude Code attribution line at the very
top, then this plan (or a summary) including the "known limits" section, then `Closes #480`.
Labels `bug` and `inherent-stability`; assignee `JackKelly`. Then a fresh adversarial review of the
PR before it reaches Jack. Never merge.
