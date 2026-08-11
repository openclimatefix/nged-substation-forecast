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

### `conf/known_dead_time_series.yaml` — new file

A bare YAML list of `time_series_id`s, with the reason for each as a comment:

```yaml
# Time series the power_data_is_fresh check ignores, because we know they are dead.
# Delete a line to start warning about that series again.
- 23  # PV site off since 2026-07-04 (James, NGED).
- 33  # Site monitor broken (James, NGED).
```

The config is a list of integers, so it gets a list of integers: no pydantic model, no wrapper key,
no id-to-reason mapping. The reason a series is silenced matters to the human deciding whether the
entry is still true, and that human is reading this file — so a YAML comment is the right home for
it, and nothing has to carry reason strings into the evaluator or the UI.

`Dockerfile` already does `COPY conf/ conf/`, so the deployment picks the file up with no change.

### `packages/contracts/src/contracts/settings.py` — where to find it

Add `known_dead_time_series_path: Path`, defaulting to
`PROJECT_ROOT / "conf" / "known_dead_time_series.yaml"`, exactly like `cv_config_path`. Every other
`conf/` path is a `Settings` field, and the `KNOWN_DEAD_TIME_SERIES_PATH` env var that follows from
`env_prefix=""` is the lever the existing `env` test fixture already uses for every other path.

### `src/nged_substation_forecast/defs/checks.py` — the behaviour

- **`_read_known_dead_ids(settings)`** — new, returning `tuple[int, ...]`. An absent file is an
  empty list, not an error. `yaml.safe_load(...) or ()` then `int()` each entry. It catches
  `OSError`, `ValueError`, `TypeError` and `yaml.YAMLError`, logs the traceback, and degrades to
  silencing nothing.

  That handler is the one piece of machinery this feature adds purely for robustness, and it earns
  its place: silencing is an optional convenience, and an optional convenience must not be able to
  blind the check's primary signal. Without it, a malformed file costs the whole freshness report —
  every late series, in the hour they went late — because the catch-all above returns "could not
  evaluate". `_read_promoted_model_facts` degrades an unreadable `meta.json` for exactly this
  reason. The degradation needs no vocabulary of its own: with nothing silenced, the dead series
  reappear in `late` and the check goes yellow about them, so a broken list announces itself
  through the very noise it was suppressing.

- **`evaluate_power_freshness`** gains `silenced_ids: Collection[int] = ()`. It stays pure. After
  `late` is built as it is today, drop the rows whose `time_series_id` is in `silenced_ids`, and
  count `n_stale` and `n_never` from what survives — so the description, the metadata counts and the
  Sentry tags all describe what we are actually watching. `n_series_total` keeps its present
  meaning, every id we know of, so a number Sentry already reports does not quietly change
  definition.

  Doing the filtering *here*, rather than in `_to_asset_check_result`, is the whole point:
  `report_power_freshness` is handed this same result, so Sentry inherits the silencing for free and
  a fully-silenced feed produces `is_healthy == True`, which is the existing no-op gate. Filtering
  later would leave the hourly Sentry warning firing about the dead series forever.

  Resurrection is computed here too, from the cutoff the function already has: a silenced id whose
  `last_time` is at or after the cutoff is alive again. One expression —
  `sorted(set(silenced_ids) & set(coverage.filter(pl.col("last_time") >= cutoff)["time_series_id"]))`
  — not a new concept. It is deliberately not "a silenced id that is absent from `late`", because
  that would also match an id that has been dropped from the roster and has no rows on disk.

- **`PowerFreshnessResult`** gains exactly two fields, both plain tuples: `silenced_ids` (the
  configured list, echoed verbatim) and `resurrected_ids`. No frame of silenced rows: `hours_late`
  for a series everyone already knows is dead is the one number about it that carries no
  information, and the issue asks which ids are ignored, not how dead each one is. Echoing the
  configured list rather than the rows actually withheld also makes a typo'd id visible — it is
  reported as ignored while silencing nothing.

  `is_healthy` keeps its exact present meaning, "no series is late", because it is the gate
  `report_power_freshness` reads and a resurrection is not a stale series. Its docstring says so.

- **`_to_asset_check_result`** does three things more than it does today:
  - `passed = result.is_healthy and result.n_series_total > 0 and not result.resurrected_ids` — a
    resurrection makes the check yellow, which is the "alert me when they come back" requirement.
  - Appends `Ignoring 2 known-dead time series: 23, 33.` to the description whenever anything is
    silenced. Uncapped, unlike every other listing in this module: the other caps guard against a
    machine-generated explosion (a whole-feed stall puts 2,500 rows in `late` with no human
    involved), while this list is bounded by what somebody typed into a git-tracked file.
  - Appends `Series 23 has reported again — remove it from conf/known_dead_time_series.yaml.` when
    `resurrected_ids` is non-empty.

  Two new metadata keys: `n_silenced`, and `silenced_time_series_ids` as a string of the list,
  following the `missing_time_series_ids` precedent. Resurrections get no metadata key of their own:
  they already flip `passed` and are named in the description, and the count is zero in every run
  but the rare one.

- **`_check_power_data_freshness`** passes `_read_known_dead_ids(settings)` into the evaluator.

- **`power_data_is_fresh`** and **`_late_table_metadata`** are unchanged.

## Design-philosophy check

This is production code, so it degrades rather than raises.

- **Rules 6 and 7 hold.** The check stays `AssetCheckSeverity.WARN` with `blocking=False`, and its
  whole body stays under the existing `BaseException` catch-all. Nothing added can raise past that
  guard: the only new I/O is one small local file read, and it has its own handler inside the
  catch-all.
- **A malformed dead-list cannot silence anything.** This is the hazard the issue's mechanism
  introduces, and the fallback is chosen so that it fails *towards* warning. It cannot fail closed
  like #480, and it cannot fail silent either.
- **This is the one per-read fallback `power_data_is_fresh` gets**, against a module docstring that
  says it salvages nothing below the catch-all. The reason that rule exists is that its only other
  candidate fallback — "roster unknown" — would render a corrupt roster as a green tick. The
  dead-list fallback is safe in the opposite direction: degrading it can only add warnings, never
  remove them. The module docstring and
  [Production deployment](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/)
  both say this explicitly rather than leaving the exception looking like drift.
- **Rule 2 (strict about malformed) is honoured where it belongs.** The shipped file is parsed by a
  test in CI, so a malformed list is rejected before it can ship; the production caller degrades
  because by then rejecting costs more than it buys.
- **H1 / T1.1.** A silenced dead series is a series that no longer demands a human glance every day,
  which is the claim
  [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
  makes. Editing the YAML when a series dies or returns is an intervention and belongs in the
  [intervention log](https://openclimatefix.github.io/nged-substation-forecast/live_service/intervention-log/)
  under `routine-ops`; the operations page will say so.

## Tests

All in `tests/test_checks.py`. Every assertion fails on `main` today, because `main` has no
silencing at all.

**First, a hazard to fix before adding anything.** `Settings.known_dead_time_series_path` defaults
to the real repo file, so without this the existing end-to-end tests would silently start silencing
23 and 33. The `env` fixture must point `KNOWN_DEAD_TIME_SERIES_PATH` at an absent path under
`tmp_path`, making silencing opt-in per test. Nothing collides today (the existing tests use ids 1,
2 and 99), which is exactly why this would go unnoticed.

| Test | Assertion that fails on `main` |
|---|---|
| `test_silenced_series_are_withheld_and_others_still_warn` | one fixture: 23 stale and silenced, 33 never-reported and silenced, 7 stale and not silenced → `n_stale == 1`, `n_never == 0`, `late` holds only 7, `n_series_total == 3` |
| `test_a_silenced_series_that_reports_again_fails_the_check` | 23 fresh and silenced → `resurrected_ids == (23,)`, `_to_asset_check_result(...).passed is False`, description names 23 and the YAML path |
| `test_the_check_names_the_silenced_ids_when_everything_is_fresh` | a green result whose description still says `Ignoring 2` and whose metadata carries `n_silenced == 2` and `silenced_time_series_ids` |
| `test_a_fully_silenced_feed_hands_sentry_a_healthy_result` | extends the existing `report_power_freshness` monkeypatch test: the captured `PowerFreshnessResult.is_healthy is True`, so no Sentry warning is sent |
| `test_power_data_is_fresh_silences_the_configured_dead_series` | end-to-end through `Settings`: YAML in `tmp_path` via `KNOWN_DEAD_TIME_SERIES_PATH`, real Delta table and roster, check passes and names the ignored ids |
| `test_power_data_is_fresh_warns_loudly_when_the_dead_list_is_malformed` | garbage YAML → the check does not raise, nothing is silenced, and the dead series is reported in `late` |
| `test_the_shipped_known_dead_list_parses` | `_read_known_dead_ids` over the real `conf/known_dead_time_series.yaml` returns a non-empty tuple of ints, so a hand-edit that breaks the file fails CI rather than reaching production. Precedent: `test_load_cv_config_reads_canonical_yaml` |

The fourth test is the requirement "stop telling me about 23 every day", asserted at the point where
it is actually delivered. It lives here rather than in `tests/test_sentry.py` so it does not collide
with #488's session.

Note for whoever writes these: the check's catch-all swallows `pytest.fail`, so a "must not be
called" sentinel inside the check body is useless — assert after the call. The existing tests say
the same thing.

## Docs to update

- **`src/nged_substation_forecast/defs/checks.py` module docstring** — a short paragraph on
  silencing, and a clause on the "salvages nothing below the catch-all" sentence naming the one
  fallback and why its direction is safe.
- **`docs/live_service/operations.md`**, "Reading the freshness check" — how to silence a series
  (edit the YAML, commit, redeploy), how to un-silence one, what `Ignoring N known-dead…` and
  `has reported again` mean, and that a resurrection stays yellow until the file is edited. Also
  that the edit is an intervention worth logging. This paragraph is the real deliverable for an
  operator, and the page already has the hook for it: "A handful of persistently-late series is
  usually a decommissioned or renamed substation".
- **`docs/architecture/production-deployment.md`**, the `power_data_is_fresh` section — why the
  list is a git-tracked file rather than mutable state, why the filtering sits in the pure evaluator
  so Sentry inherits it, and the one-clause correction to "salvages nothing".
- **`README.md` / `CLAUDE.md`** — no change; neither enumerates `conf/`.

This issue does not complete a roadmap item, so there is no ship-time triage of an "Implementation
details" section.

## Verification commands

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check
uv run pytest tests/test_checks.py
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
3. **Should `live_forecasts_are_healthy` also ignore dead series?** Its `missing_time_series_ids`
   compares the promoted model's trained population against what the slot forecast, and a dead
   series is still forecast from its other features, so it does not go missing. *Recommendation:*
   leave it alone; revisit if a dead series does start dropping out of slots.

## Findings from review, and what happened to each

### First pass — is there a simpler way? (accepted)

- **No pydantic model and no `contracts` change.** The config is a list of integers, so `CvConfig`'s
  precedent does not carry: a bare YAML list parsed in `checks.py` does the same job, and the
  id-to-reason mapping the first draft proposed carried strings that were never surfaced anywhere.
  Two files and four loader tests removed.
- **`silenced` is a tuple of ids, not a DataFrame.** Dropped the silenced-series metadata table,
  its `n_silenced_listed` count, and the second caller for `_late_table_metadata`.
- **Dropped `_MAX_SILENCED_SERIES_LISTED`.** Every other cap here guards a machine-generated
  listing; this one is bounded by a human commit.
- **Metadata down from five new keys to two.** Both resurrection keys go: the description names the
  ids and `passed` already flips.
- **Dropped `known_dead_unreadable` from the result and its description branch.** An unreadable
  list announces itself by the dead series reappearing as late, so the flag bought a fourth branch
  and no new information.
- **Tests merged from eleven to seven**, and the reviewer's `env`-fixture hazard — the default path
  resolving to the real repo file during tests — is now fixed explicitly rather than by luck.
- **Added a CI test that parses the shipped file**, on the `test_load_cv_config_reads_canonical_yaml`
  precedent.

### First pass — rejected

- **"Drop the runtime fallback entirely; let a malformed list hit the catch-all, since CI parses the
  shipped file."** Rejected: an optional convenience must not be able to blind the check's primary
  signal, and `_read_promoted_model_facts` sets the precedent for degrading a small file read
  in-place. The CI test is worth having and has been added, but it guards the file we ship, not the
  file that is there at runtime. The reviewer's supporting point — that the catch-all route would
  reach Sentry via `report_check_degradation` while the fallback does not — is real, and is now
  covered by note 2 for #488.
- **"Make the path a module constant instead of a `Settings` field."** Rejected: every other `conf/`
  path is a `Settings` field, and the env-var lever is how the existing `env` test fixture redirects
  every other path. A module constant would need attribute monkeypatching instead.
- **"Compute resurrection as `silenced_ids` minus `late` instead of against the cutoff."** Rejected
  by the reviewer's own analysis, and the plan now says why in the code description.
