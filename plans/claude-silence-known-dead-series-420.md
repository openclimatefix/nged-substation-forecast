# Plan — Silence warnings for known-dead time series (#420)

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/420>
Branch: `claude/silence-known-dead-series-420`

## Verdict

Worth implementing, and worth implementing now, but **only the file-backed half** of what the
issue's comments describe.

The issue asks for three behaviours, and all three are satisfiable inside the vocabulary
`power_data_is_fresh` already has:

1. We are still told which ids are being ignored — every hour, in the check's description and
   metadata, green or yellow.
2. We are alerted if a silenced series comes back — the check turns yellow and names the id.
3. Any other series going bad still warns exactly as loudly as it does today.

Series 23 has been dead since 4 July 2026, so the hourly check has been yellow for about five
weeks. That is the warning-fatigue failure the
[provider channel](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#three-audiences-three-channels)
cannot survive, and it is happening now.

### Departures from the issue and its comments

**The list lives in `conf/known_dead_time_series.yaml`, not in the Dagster UI.** Jack's most recent
comment describes the ideal: an operator adds and removes ids from the Dagster UI, and a returning
series is removed from the list automatically. That version needs operator-writable persistent
state plus a UI affordance to write it, and the auto-removal puts a *write* inside the one code path
that must never fail (rule 7). This plan builds the version Jack's earlier comment proposed — YAML
in `conf/`, reviewable in git, travelling with the code because a series is dead whether we run on
a laptop or on AWS — and leaves the UI-editable version to a follow-up issue, sequenced after
`power_forecast_warnings` (#439, #441) and `asset_health_history` (#442) so it is written against
the real warning vocabulary rather than a guess at it. See *Open questions*.

**A returning series is reported, not auto-removed.** The check names the resurrected id and stays
yellow until a human deletes the line from the YAML. The yellow is the reminder; it clears itself
the moment the file is edited. Editing `conf/` from inside a production asset check is the wrong
trade at every level: it is a write on the warning path, and it silently rewrites a git-tracked file
that a deployment ships read-only in a container image.

**No Sentry change in this issue.** #488 owns the Sentry event shape. Two consequences are recorded
below as notes for that issue.

## What changes, file by file

### `packages/contracts/src/contracts/config_schemas.py` — the list and its loader

Add a `KnownDeadTimeSeries` pydantic model beside `CvConfig`, with one field:

```yaml
known_dead:
  23: "PV site off since 2026-07-04 (James, NGED)."
  33: "Site monitor broken (James, NGED)."
```

`known_dead: dict[int, str]` — id to the reason it is silenced. The reason is for the human who
later has to decide whether the entry is still true; it is not surfaced in the Dagster UI (see
*Rejected*). `extra="forbid"` so a typo'd top-level key is rejected rather than ignored, matching
#512's direction of travel.

Add `load_known_dead_time_series(path: Path) -> KnownDeadTimeSeries`, mirroring `load_cv_config`:
`yaml.safe_load` then `model_validate`, with `or {}` so an empty file loads as an empty mapping
rather than validating `None`. It **raises** on a malformed file — this is a config boundary, and
rule 2 is strict about malformed input. The degradation happens one level up, in the check.

### `packages/contracts/src/contracts/settings.py` — where to find it

Add `known_dead_time_series_path: Path`, defaulting to
`PROJECT_ROOT / "conf" / "known_dead_time_series.yaml"`, exactly like `cv_config_path`. The env var
`KNOWN_DEAD_TIME_SERIES_PATH` follows from `env_prefix=""`, which is what the tests use to point at
a `tmp_path` file.

### `conf/known_dead_time_series.yaml` — new file

Ships with 23 and 33 and their reasons, plus a header comment saying what the file does, that
removing a line un-silences the series, and that the freshness check turns yellow when a listed
series reports again. `Dockerfile` already does `COPY conf/ conf/`, so the deployment picks it up
with no change.

### `src/nged_substation_forecast/defs/checks.py` — the behaviour

- **`evaluate_power_freshness`** gains `silenced_ids: Collection[int] = ()`. It stays pure. After
  `late` is built as it is today, split it: rows whose `time_series_id` is in `silenced_ids` move to
  a new `silenced` frame, the rest stay in `late`. `n_stale` and `n_never` count the rows that
  survive the split, so the description, the metadata counts and the Sentry tags all describe what
  we are actually watching. `n_series_total` keeps its present meaning — every id we know of — so a
  number that Sentry already reports does not quietly change definition.

  Doing the filtering *here*, rather than in `_to_asset_check_result`, is the whole point:
  `report_power_freshness` is handed this same result, so Sentry inherits the silencing for free and
  a fully-silenced feed produces `is_healthy == True`, which is the existing no-op gate. Filtering
  later would leave the hourly Sentry warning firing about the dead series forever.

  Resurrection is computed here too: a silenced id whose `last_time` is at or after the cutoff is
  alive again.

- **`PowerFreshnessResult`** gains three fields: `silenced: pl.DataFrame` (the withheld rows, same
  four columns as `late`), `resurrected_ids: tuple[int, ...]`, and `known_dead_unreadable: bool`.
  Plus an `n_silenced` property. `is_healthy` keeps its exact present meaning — "no series is
  late" — because it is the gate `report_power_freshness` reads, and a resurrection is not a stale
  series. Its docstring says so.

- **`_late_table_metadata`** is unchanged and gains a second caller: the silenced frame has the same
  four columns, so it renders through the same function and the same `_LATE_TABLE_SCHEMA`. Its
  docstring is updated to name both callers.

- **`_to_asset_check_result`** does four things more than it does today:
  - `passed = result.is_healthy and result.n_series_total > 0 and not result.resurrected_ids` — a
    resurrection makes the check yellow, which is the "alert me when they come back" requirement.
  - Appends `Ignoring N known-dead time series: 23, 33.` to the description whenever anything is
    silenced, capped at a new `_MAX_SILENCED_SERIES_LISTED` (20, for the same reason the other
    listings are capped: V2 is ~2,500 series and this lands in the event log hourly).
  - Appends `Series 23 has reported again — remove it from conf/known_dead_time_series.yaml.` when
    `resurrected_ids` is non-empty.
  - Appends `The known-dead list could not be read, so nothing is silenced.` when
    `known_dead_unreadable`.

  New metadata keys: `n_silenced`, `n_silenced_listed`, `silenced_time_series` (the table),
  `n_resurrected`, `resurrected_time_series_ids`. The last two follow the `missing_time_series_ids`
  precedent: a string of the list, capped, with its own count.

- **`_check_power_data_freshness`** reads the list through a new `_read_known_dead_ids(settings)`
  which returns `dict[int, str] | None`. `None` means "could not read": an absent file is *not*
  unreadable, it is an empty list. It catches `OSError`, `ValueError` (pydantic's
  `ValidationError` derives from it) and `yaml.YAMLError`, logs the traceback, and degrades to
  silencing nothing — mirroring `_read_promoted_model_facts`.

- **`power_data_is_fresh`** itself is unchanged: same catch-all, same WARN, same `blocking=False`.

## Design-philosophy check

This is production code, so it degrades rather than raises.

- **Rules 6 and 7 hold.** The check stays `AssetCheckSeverity.WARN` with `blocking=False`, and its
  whole body stays under the existing `BaseException` catch-all. Nothing added can raise past that
  guard: the only new I/O is one small local file read, and it has its own handler above the
  catch-all.
- **A malformed dead-list cannot silence anything.** This is the hazard the issue's mechanism
  introduces, and the fallback is chosen so that it fails *towards* warning: an unreadable list
  silences nothing, so every dead series reappears in `late` and the check goes yellow and says
  why. A broken list is therefore self-announcing — it cannot fail closed like #480, and it cannot
  fail silent either.
- **This is the one per-read fallback `power_data_is_fresh` gets**, against a module docstring that
  says it salvages nothing below the catch-all. The reason that rule exists is that its only other
  candidate fallback — "roster unknown" — would render a corrupt roster as a green tick. The
  dead-list fallback is safe in the opposite direction: degrading it can only add warnings, never
  remove them. The module docstring and
  [Production deployment](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/)
  both say this explicitly rather than leaving the exception looking like drift.
- **Rule 2 (strict about malformed) is honoured where it belongs.** The loader raises; the
  production caller degrades. Being strict at the boundary and liberal at the point of use is the
  same split `_read_promoted_model_facts` already makes.
- **H1 / T1.1.** A silenced dead series is a series that no longer demands a human glance every day,
  which is the claim
  [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
  makes. Editing the YAML when a series dies or returns is an intervention and belongs in the
  [intervention log](https://openclimatefix.github.io/nged-substation-forecast/live_service/intervention-log/)
  under `routine-ops`; the operations page will say so.

## Tests

Every assertion below fails on `main` today, because `main` has no silencing at all.

**`tests/test_checks.py`**, a new section beside the existing freshness tests:

| Test | Assertion that fails on `main` |
|---|---|
| `test_a_silenced_stale_series_is_withheld` | id 23 stale, silenced → `n_late == 0`, `is_healthy`, `silenced.height == 1` |
| `test_a_silenced_never_reported_series_is_withheld` | 33 in the roster with no data, silenced → `n_never == 0` |
| `test_an_unsilenced_series_still_warns_beside_a_silenced_one` | 23 silenced, 7 stale → `n_stale == 1` and `late` holds only 7 |
| `test_a_silenced_series_that_reports_again_is_resurrected` | 23 fresh and silenced → `resurrected_ids == (23,)` |
| `test_a_resurrection_makes_the_check_fail_and_names_the_id` | `_to_asset_check_result(...).passed is False`, description names 23 and the YAML path |
| `test_the_check_names_the_silenced_ids_when_everything_is_fresh` | green result whose description still says `Ignoring 2` and whose metadata carries `n_silenced == 2` and the `silenced_time_series` table |
| `test_silencing_does_not_change_n_series_total` | `n_series_total` counts the silenced ids too |
| `test_power_data_is_fresh_silences_the_configured_dead_series` | end-to-end through `Settings`: YAML in `tmp_path` via `KNOWN_DEAD_TIME_SERIES_PATH`, real Delta + roster, check passes |
| `test_power_data_is_fresh_warns_loudly_when_the_dead_list_is_malformed` | garbage YAML → the check does not raise, `n_silenced == 0`, the dead series is in `late`, the description says the list could not be read |
| `test_power_data_is_fresh_with_no_dead_list_behaves_as_before` | absent file → identical result to today, and no "could not be read" text |
| `test_a_fully_silenced_feed_hands_sentry_a_healthy_result` | extends the existing `report_power_freshness` monkeypatch test: the captured `PowerFreshnessResult.is_healthy is True`, so no Sentry warning is sent |

The last one is the requirement "stop telling me about 23 every day", asserted at the point where it
is actually delivered. It lives in `tests/test_checks.py` rather than `tests/test_sentry.py` so it
does not collide with #488's session.

**`packages/contracts/tests/test_config_schemas.py`**: the loader round-trips a valid file; an
empty file loads as an empty mapping; an unknown top-level key raises; a non-integer id raises.

Note for whoever writes these: the check's catch-all swallows `pytest.fail`, so a "must not be
called" sentinel inside the check body is useless — assert after the call. The existing tests say
the same thing.

## Docs to update

- **`src/nged_substation_forecast/defs/checks.py` module docstring** — a paragraph on silencing, and
  an amendment to the "salvages nothing below the catch-all" paragraph naming the one fallback and
  why its direction is safe.
- **`docs/live_service/operations.md`**, "Reading the freshness check" — how to silence a series
  (edit the YAML, commit, redeploy), how to un-silence one, what `Ignoring N known-dead…`,
  `has reported again` and `could not be read` each mean, and that a resurrection stays yellow until
  the file is edited. Also that the edit is an intervention worth logging.
- **`docs/architecture/production-deployment.md`**, the `power_data_is_fresh` section — why the list
  is a git-tracked file rather than mutable state, why the filtering sits in the pure evaluator so
  Sentry inherits it, and why an unreadable list un-silences everything.
- **`README.md` / `CLAUDE.md`** — no change; neither enumerates `conf/`.

This issue does not complete a roadmap item, so there is no ship-time triage of an "Implementation
details" section.

## Verification commands

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check
uv run pytest tests/test_checks.py packages/contracts/tests/test_config_schemas.py
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

No network-gated tests are involved.

## Notes for #488 (Sentry event shape) — do not implement here

1. **A resurrection should send a "happy" Sentry event.** Jack asked for one. It cannot be added
   here without editing `_sentry.py`: `report_power_freshness` is gated on `result.is_healthy`, and
   a resurrection leaves the feed healthy by that definition. `PowerFreshnessResult.resurrected_ids`
   is the field to read.
2. **The freshness warning event should say how many series are silenced.** After this change,
   Sentry sees a smaller `n_late` and no dead-series rows at all, with nothing in the event saying
   the silencing is on. Adding `n_silenced` to the `power_freshness` context and the message closes
   that gap. In-band the operator does still see it, in the Dagster description every hour.

## Risks and open questions

1. **Is James's "TimeSeriesInstance 23" our `time_series_id` 23?** `time_series_id` is documented as
   "Provided by NGED", so almost certainly yes, but seeding the file with a wrong id silences a
   healthy series and hides a real fault. *Recommendation:* ship the file with 23 and 33 as planned,
   and check the ids against the live freshness check's late-series table before merging — if 23 and
   33 are the two that have been yellow since July, the mapping is confirmed by the data.
2. **Do you want the follow-up issue for the UI-editable version?** *Recommendation:* yes, opened
   after this merges, blocked by #439/#441/#442, describing the operator-writable list plus
   auto-removal on resurrection. I can draft it once you have approved this plan.
3. **Should `live_forecasts_are_healthy` also ignore dead series?** Its
   `missing_time_series_ids` compares the promoted model's trained population against what the slot
   forecast, and a dead series is still forecast from its other features, so it does not go missing.
   *Recommendation:* leave it alone; revisit if a dead series does start dropping out of slots.

## Findings from review, and what happened to each

*(filled in by the two adversarial review passes)*
