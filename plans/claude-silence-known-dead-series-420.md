# Plan — Silence warnings for known-dead time series (#420)

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/420>
Branch: `claude/silence-known-dead-series-420`

## Verdict

Worth implementing, and worth implementing now, but **only the "list lives with the code" half** of
what the issue's comments describe.

The issue asks for three behaviours, and all three are satisfiable inside the vocabulary
`power_data_is_fresh` already has:

1. We are still told which ids are being ignored — every hour, in the check's description and
   metadata, green or yellow.
2. We are told when a silenced series comes back — the check turns yellow and names the id. That is
   Dagster's Checks view, not Sentry: a Sentry "happy message" needs `_sentry.py`, which #488 owns.
3. Any other series going bad still warns exactly as loudly as it does today.

Series 23 has been dead since 4 July 2026, so the hourly check has been yellow for about five
weeks. That is the warning-fatigue failure the
[provider channel](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#three-audiences-three-channels)
cannot survive, and it is happening now.

The whole change is a `Final` tuple, one keyword-only parameter, three lines inside
`evaluate_power_freshness`, two description clauses and five tests.

### Departures from the issue and its comments

**The list is a `Final` tuple in `checks.py`, not `conf/known_dead_time_series.yaml`.** This
departs from Jack's comment — "the list of dead timeseries should live with the code … So maybe
YAML in `conf/`?" — on the *file*, while agreeing with it on the *principle*: a module constant
lives with the code more literally than `conf/` does. The question that comment was answering was
code versus `.env`, and both forms answer it the same way. What tips it is what the YAML costs:
every mis-typing that a Python literal turns into a ruff, `ty` or import-time error becomes, in
YAML, a runtime state that has to be guarded, degraded, documented and tested — and `"2333"` parses
happily into silencing series 2, 3, 3 and 3. Neither form is operator-editable in any case: both
need a commit, an image rebuild and a redeploy. This is open question 1, and reverting to YAML is a
contained change if Jack prefers it.

**One list, not the `known late` plus `known missing` pair the same comment suggests.** The
distinction is already carried, per series and per hour, by the `status` column the check computes:
a silenced id shows up as `stale` or `never` depending on what the data says. Two lists would make
the operator predict which bucket a dead series will land in, and be wrong the first time a
never-reported series briefly reports.

**The list is not editable from the Dagster UI, and a returning series is not auto-removed.** Jack's
most recent comment describes that ideal. It needs operator-writable persistent state plus a UI
affordance to write it, and the auto-removal puts a *write* inside the one code path that must never
fail (rule 7). Here the check names the resurrected id, and stays yellow for as long as that series
has data newer than the 24-hour threshold. Be precise about what that does and does not promise:
`resurrected_ids` is recomputed from scratch every hour with no remembered state, so a series that
reports for a few hours and dies again shows yellow only during that window and then goes green with
the silencing still in force. A durable "this came back at least once" signal needs remembered
state, which is `asset_health_history` (#442) and the follow-up issue, not this one. The plan and
the operations page both say the transient version, because it is the true one.

**Implemented now, rather than after #439, #441 and #442.** Jack's first comment says this is
"worth sequencing downstream of those three so the silencing rules are written against the real
warning vocabulary rather than a guess at it". That reasoning applies to the UI-editable version,
which invents vocabulary; this version invents none — it reuses the check's existing `stale` /
`never` / `late` words and adds a constant. Against a five-week-old permanent yellow, waiting for
three unstarted issues is the worse trade. The follow-up issue keeps the sequencing. See open
question 2.

**No Sentry change in this issue.** #488 owns the Sentry event shape. Two consequences are recorded
below as notes for that issue.

## What changes, file by file

Only `src/nged_substation_forecast/defs/checks.py`, its tests, and three docs pages. No new file, no
`Settings` field, no `contracts` change, no `yaml` import in a production check module.

### `src/nged_substation_forecast/defs/checks.py`

- **`_KNOWN_DEAD_TIME_SERIES_IDS: Final[tuple[int, ...]] = (23, 33)`** — new, beside
  `_POWER_DATA_STALENESS_THRESHOLD`. Its docstring carries the reason for each id (23: PV site off
  since 2026-07-04; 33: site monitor broken; both reported by James at NGED), says that deleting an
  entry starts the warnings again, and says the check turns yellow by itself if a listed series
  reports data again.

- **`evaluate_power_freshness`** gains a keyword-only `silenced_ids: Collection[int] = ()`. It stays
  pure. Three lines go in immediately after `cutoff` is computed, before `stale` is built:
  - `resurrected_ids` — the silenced ids whose `last_time` is at or after the cutoff, sorted.
  - `coverage` is filtered to drop the silenced ids.
  - `roster_ids`, when it is not `None`, is filtered the same way.

  Everything downstream then describes the watched population with no further edit: `stale`,
  `never`, `late`, `n_stale`, `n_never`, `n_series_total`, the description, the metadata counts and
  the whole Sentry payload. Verified on the installed Polars (1.43.2): `is_in` takes a plain Python
  list against an `Int32` column on both the expression and the `Series`, and an empty list is a
  no-op, so the `silenced_ids=()` default needs no special case.

  Filtering *here*, in the pure evaluator, is the structural point of the change:
  `report_power_freshness` is handed this same result, so Sentry inherits the silencing for free and
  a fully-silenced feed produces `is_healthy == True`, which is the existing no-op gate. Filtering
  in `_to_asset_check_result` instead would leave the hourly Sentry warning firing about the dead
  series forever.

  Resurrection is deliberately *not* "a silenced id absent from `late`", for two reasons. Under
  this design there is no unfiltered `late` to subtract from: the silenced ids are gone before
  `stale` and `never` are built, so *every* silenced id is absent from `late` and the subtraction
  would report them all as resurrected. Restoring an unfiltered `late` to subtract from would mean
  going back to filtering at the bottom of the function, which is the structure this design
  replaced. And even in that structure the subtraction is wrong for a silenced id that was never in
  the roster and has no rows on disk — a typo in the constant, or an id NGED has never published. It
  is in neither `coverage` nor `roster_ids`, so it is neither stale nor never-reported, so it is
  absent from `late`, so subtraction calls it resurrected. A mistyped `int` is valid Python that
  `ty` cannot catch, so that case is reachable in a way a malformed config file is not.

  `stale` uses `last_time < cutoff` and resurrection uses `>= cutoff`, so the two are complementary
  over `coverage` — a series is stale, fresh, or has no rows at all, and only the middle one is a
  resurrection.

- **`PowerFreshnessResult`** gains two fields, both defaulted so the three existing construction
  sites keep working untouched (`tests/test_checks.py`, `tests/test_sentry.py`, and the operator's
  copy-pasteable smoke script in `docs/live_service/sentry.md` — all three pass every field by
  keyword): `silenced_ids: tuple[int, ...] = ()` and `resurrected_ids: tuple[int, ...] = ()`.

  `is_healthy` keeps its exact present meaning, "no series is late", because it is the gate
  `report_power_freshness` reads and a resurrection is not a stale series. Its docstring says so.

  `n_series_total` changes meaning, from "every id we know of" to "every id we are watching". That
  is what makes the description and the Sentry denominator true with no arithmetic. Its docstring
  and its reader **in `checks.py`** say the new meaning; its two other readers are in `_sentry.py`,
  which this issue may not touch, so they are note 2 for #488.

- **`_to_asset_check_result`** does three things more than it does today:
  - `passed = result.is_healthy and result.n_series_total > 0 and not result.resurrected_ids` — a
    resurrection makes the check yellow, which is the "tell me when they come back" requirement.
  - Appends `Ignoring 2 known-dead time series: 23, 33.` to the description on **every** branch
    whenever `silenced_ids` is non-empty; and appends `Series 23 has reported again — remove it
    from _KNOWN_DEAD_TIME_SERIES_IDS in defs/checks.py.` when `resurrected_ids` is non-empty. The
    ignoring clause is uncapped, unlike every other listing in this module: the other caps guard
    against a machine-generated explosion (a whole-feed stall puts 2,500 rows in `late` with no
    human involved), while this one is bounded by what somebody typed into a source file.
  - **Splits the `n_series_total == 0` branch in two**, because the redefinition above gives that
    guard a second meaning. `n_series_total == 0` with nothing silenced still means "No power data
    on disk yet."; with something silenced it means every id we know of is on the dead list, which
    is a different and much louder statement, so it gets its own sentence — `Every known time
    series is silenced as known-dead: 23, 33.` Both keep `passed=False`: watching nothing is not
    healthy. This is the one place the meaning change costs an extra branch rather than falling out
    for free, and it is unreachable at V1 (2 silenced of 32) but trivially reachable on a laptop
    with a small local table.

  Two new metadata keys, emitted on every run so each keeps one type and stays plottable — the rule
  the existing `n_late_listed` follows: `n_silenced`, and `silenced_time_series_ids` as a string of
  the list, following the `missing_time_series_ids` precedent. Resurrections get no key of their
  own: they already flip `passed` and are named in the description, and the count is zero in every
  run but the rare one.

- **`_check_power_data_freshness`** passes `silenced_ids=_KNOWN_DEAD_TIME_SERIES_IDS`. At the call
  site, not as the parameter's default value — a default is bound at import and
  `monkeypatch.setattr` on the constant would not reach it, so the end-to-end test below would
  silently exercise the real ids.

- **`power_data_is_fresh`** and **`_late_table_metadata`** are unchanged.

Two notes for the implementer. `docs/architecture/code-style.md` now requires keyword arguments at
call sites, which is why the new parameter is keyword-only and why the call above names it. And the
parameter is a `Collection[int]` while the result field is a `tuple[int, ...]`, so the constructor
needs `tuple(silenced_ids)`.

## Design-philosophy check

This is production code, so it degrades rather than raises.

- **Rules 6 and 7 hold.** The check stays `AssetCheckSeverity.WARN` with `blocking=False`, and its
  whole body stays under the existing `BaseException` catch-all. The silencing input is a module
  constant, so there is no I/O, no parse and nothing to degrade from — the mechanism has no runtime
  failure mode that a well-typed constant can reach. (It is not *unconditionally* unraisable: a
  non-`int` entry would make `is_in` raise `InvalidOperationError`. `Final[tuple[int, ...]]` plus
  `ty`, which runs in both pre-commit and CI, is the gate — and even then the catch-all holds, so
  the worst case is a degraded check, never a failed run.)
- **The module docstring's "salvages nothing below the catch-all" stays true as written.** The
  earlier YAML-based draft had to carve an exception into it in two places; this one does not.
- **Rule 2 (strict about malformed) is honoured statically.** A mistyped id is a ruff or `ty` error
  before it is anything else.
- **`live_forecasts_are_healthy` is left alone.** Its `missing_time_series_ids` compares the promoted
  model's trained population against what the slot forecast, and a dead series is still forecast
  from its other features, so it does not go missing.
- **H1 / T1.1.** A silenced dead series is a series that no longer demands a human glance every day,
  which is the claim
  [H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself)
  makes. Editing the list when a series dies or returns is an intervention and belongs in the
  [intervention log](https://openclimatefix.github.io/nged-substation-forecast/live_service/intervention-log/)
  under `routine-ops`; the operations page will say so.

## Tests

All in `tests/test_checks.py`. Seven tests and one added assertion; every one fails on `main`.

| Test | What fails on `main` |
|---|---|
| `test_silenced_series_are_withheld_and_others_still_warn` | one fixture, stated in full: 7 stale and not silenced, 1 fresh and not silenced, 23 stale and silenced, 33 never-reported (roster-only) and silenced, 404 silenced but in neither roster nor coverage. Asserts `n_stale == 1`, `n_never == 0`, `late` holds only 7, `resurrected_ids == ()` and **`n_series_total == 2`** — the watched population `{1, 7}`. That last number is the assertion that tells filtering the *inputs* apart from filtering `late`, which answers 4. Id 404 is a silenced id that was never in the roster — a typo in the constant, or an id NGED has never published — and asserting it is not in `resurrected_ids` is what rules out computing resurrection as `silenced_ids` minus `late`; id 23 pins that a silenced series which is still stale is not reported as resurrected |
| `test_a_feed_whose_only_late_series_are_silenced_is_healthy` | `is_healthy is True` with every late series silenced. This is the load-bearing fact behind the issue's actual request — `report_power_freshness` returns early on a healthy result, so the daily Sentry warning about 23 stops |
| `test_a_silenced_series_that_reports_again_is_resurrected` | 23 fresh and silenced → `resurrected_ids == (23,)` and `is_healthy is True`, which is why note 1 for #488 is true. Parametrised over the cutoff boundary: `last_time` exactly at the cutoff is resurrected, a second earlier is not — the exact complement of `stale`'s `<` |
| `test_silencing_works_without_a_roster` | `roster_ids=None` plus silenced ids: the counts describe the survivors and nothing raises. A missing `if roster_ids is not None` guard would `AttributeError` into the catch-all and leave the check permanently "Could not evaluate", and no existing test pins that combination |
| `test_the_check_result_reports_silencing` | parametrised over the healthy and the late branch: the description names the ignored ids in **both** (requirement (a) matters most on a green tick), `passed is False` on resurrection with 23 named, and `n_silenced` / `silenced_time_series_ids` present |
| `test_a_wholly_silenced_roster_is_not_reported_as_an_empty_table` | `n_series_total == 0` with silencing → the description says every known series is silenced and names them, rather than "No power data on disk yet.", and `passed is False`. Pins the branch split |
| `test_power_data_is_fresh_silences_the_configured_dead_series` | end-to-end: `monkeypatch.setattr(checks, "_KNOWN_DEAD_TIME_SERIES_IDS", …)`, a real Delta table and roster, the check passes and names the ignored ids |
| *(one added assertion)* in the existing `test_power_data_is_fresh_all_current_passes` | `n_silenced == 0` and `silenced_time_series_ids == "[]"` are emitted when nothing is silenced, so the keys keep one type across runs. The same test already makes this point for `n_late_listed` |

Three notes for whoever writes these. The existing end-to-end tests use ids 1, 2 and 99, so the real
`(23, 33)` constant does not collide with any of them — but a new end-to-end test must monkeypatch
the constant rather than rely on that. Four of these fail on `main` by signature error (`TypeError`
on `silenced_ids=`, `AttributeError` from `monkeypatch.setattr` on a constant that does not exist,
`KeyError` on a new metadata key) rather than by behaviour, so their real worth is mutation
resistance — which is why `n_series_total == 2` and `n_never == 0` are worth stating exactly:
dropping the roster filter turns 33 back into a `never` row, and filtering `late` instead of the
inputs changes the total. And the check's catch-all swallows `pytest.fail`, so a "must not be
called" sentinel inside the check body is useless; assert after the call.

## Docs to update

- **`src/nged_substation_forecast/defs/checks.py`** — three prose edits inside the file:
  - The module docstring: a short paragraph on silencing — what the constant does, that filtering
    happens in the pure evaluator so the Sentry warning inherits it, and that a returning series
    turns the check yellow for as long as it keeps reporting rather than being removed
    automatically.
  - `_MAX_LATE_SERIES_IN_TABLE`'s docstring says "``n_stale`` and ``n_never_reported`` stay exact
    throughout". They now count the watched population, so say that — the same rewrite as the
    operations page below. (#544 has merged, shortening this docstring and moving the cap's
    rationale to `production-deployment.md`; the exactness claim survived the cut, so this item is
    unaffected beyond the line it sits on.) The same claim appears once more, in
    `test_never_reported_series_crowd_stale_ones_out_of_a_truncated_table`'s docstring: "the table
    can hold no stale series at all, while the counts stay exact".
  - The `NOTE` above the `stale` filter now ends "Silencing one needs an explicit record of which
    ids we have retired" — PR #540 rewrote it, so the earlier draft's item about intersecting stale
    ids with `roster_ids` no longer applies. That explicit record is what this change adds, three
    lines above the `NOTE`, so point the sentence at `_KNOWN_DEAD_TIME_SERIES_IDS`. The two are not
    quite the same population — the `NOTE` is about series NGED has retired, the constant is about
    series whose sensors are broken — and the sentence should say that one mechanism serves both.
- **`docs/live_service/operations.md`**, "Reading the freshness check" — two edits:
  - The new paragraph: how to silence a series and how to un-silence one (edit the constant, commit,
    redeploy), what `Ignoring N known-dead…` and `has reported again` mean, that the yellow lasts
    only while the returned series keeps reporting, that `n_silenced` counts *configured* ids so it
    can exceed the number actually withheld (a listed id that no longer exists silences nothing),
    and that the edit is an intervention worth logging.
  - A rewrite of "Read `n_stale` and `n_never_reported` — both always exact" and of "`n_late` … is
    the true count". All three now count the watched population, silenced series excluded. Rewrite
    the passage rather than appending a caveat.
- **`docs/architecture/production-deployment.md`**, the `power_data_is_fresh` section — why the list
  is a source constant rather than mutable state or a config file, why the filtering sits in the
  pure evaluator so Sentry inherits it, the new meaning of `n_series_total`, and the sentence "a
  green tick when every series is current and a yellow warning naming the late count when the feed
  has stalled", which now has a third cause: a silenced series reporting again.
- **`docs/live_service/sentry.md`** — no change. Its `PowerFreshnessResult` smoke script passes
  every field by keyword, so the two new fields' defaults keep it working; this is why they have
  defaults.
- **`docs/design-philosophy/inherent-stability.md`** — no change. Its ladder row "`power_data_is_fresh`
  warns and names the late series" stays true for every series that is not silenced, and the page
  describes the degradation ladder rather than this check's configuration.
- **`README.md` / `CLAUDE.md`** — no change.

This issue does not complete a roadmap item, so there is no ship-time triage of an "Implementation
details" section.

## Verification commands

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check
uv run pytest tests/test_checks.py tests/test_sentry.py
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

No network-gated tests are involved.

## Notes for #488 (Sentry event shape) — do not implement here

1. **A resurrection should send a "happy" Sentry event.** Jack asked for one. It cannot be added
   here without editing `_sentry.py`: `report_power_freshness` is gated on `result.is_healthy`, and
   a resurrection leaves the feed healthy by that definition — deliberately, because the only way to
   force an event without touching that file would send a `0/30 time series late` warning with an
   empty `late` frame. `PowerFreshnessResult.resurrected_ids` is the field to read.
2. **The freshness warning event should say that silencing is on.** After this change Sentry sees a
   smaller `n_late`, a smaller `n_series_total` and no dead-series rows, with nothing in the event
   saying why. Adding `n_silenced` and `silenced_time_series_ids` to the `power_freshness` context
   closes that gap. In-band the operator does still see it, in the Dagster description every hour.

## Risks and open questions

1. **A `Final` tuple in `checks.py`, or the `conf/` YAML file Jack's comment suggested?** The
   simplicity review argued for the constant and I have planned it that way; this is the one place
   the plan overrules a comment, so it should be a deliberate decision rather than a silent one.

   *For the constant:* the YAML's whole failure surface disappears. No `Settings` field, no
   `contracts` change, no `yaml` import in a production check module, no parse guard against
   `"2333"` silencing series 2, 3, 3 and 3, no degradation path, no exception carved into the
   module's "salvages nothing" doctrine, no CI test to prove the shipped file parses, and no test
   fixture hazard from a default path that resolves to the real repo file. About 100 lines of code,
   tests and prose, and one package, left alone. Every mistyping becomes a static error.

   *For the YAML:* it is what Jack suggested, and a non-programmer editing a two-line list in
   GitHub's web UI is a smaller ask than editing a 900-line Python module — which matters for the
   stated aim of a system NGED could operate. Against that: neither form is editable without a
   commit, an image rebuild and a redeploy, so the operator story is really the follow-up issue's
   job, and the YAML would be scaffolding thrown away when that lands.

   *Recommendation:* the constant. Reverting to YAML is contained — one file, one `Settings` field,
   one reader with a guard — if you would rather have it.
2. **Do you want the follow-up issue for the UI-editable version?** *Recommendation:* yes, opened
   after this merges, blocked by #439/#441/#442, describing the operator-writable list plus
   auto-removal on resurrection. I can draft it once you have approved this plan.
3. **Is James's "TimeSeriesInstance 23" our `time_series_id` 23?** `time_series_id` is documented as
   "Provided by NGED", so almost certainly yes, but a wrong id silences a healthy series and hides a
   real fault — the only way this change can do harm. *Recommendation:* treat it as a hard gate:
   before merging, check 23 and 33 against the live check's late-series table. If they are the two
   that have been yellow since July, the mapping is confirmed by the data.
4. **#523 touches the same function.** It splits the late-series table into separate stale and
   never-reported listings — same `_to_asset_check_result`, same metadata block, different concern.
   Neither needs the other, so whichever lands second rebases; the conflict is textual, not
   semantic. *Recommendation:* no sequencing constraint, but do not run both in the same wave.

## Findings from review, and what happened to each

Four reviews. A simplicity pass and a correctness pass on the first draft; then, after merging main
brought an updated `plan-issue` skill, a second simplicity pass with the skill's new reachability
attack and its licence to propose a different architecture — and, because that pass replaced the
mechanism, a second correctness pass over what it left behind.

### Accepted

- **The headline test asserted the *pre-change* `n_series_total`** (fourth pass). The fixture as
  written gives a watched population of one, not three — three is what `main` answers, and what an
  implementation that filtered `late` instead of the inputs would answer. The fixture is now stated
  in full and the assertion is the watched value, which makes it the test that tells the two
  implementations apart.
- **"Stays yellow until a human deletes the line" was false** (fourth pass). `resurrected_ids` is
  recomputed hourly with no remembered state, so a series that reports for a few hours and dies
  again goes yellow and then green with the silencing still on. The plan promised the durable
  version and would have written it into the operations page. Both now state the transient version,
  and the durable one is named as needing `asset_health_history` (#442).
- **The `n_series_total == 0` branch was left incoherent** (fourth pass). Redefining the quantity
  gave that guard a second meaning: with every id silenced the check would report "No power data on
  disk yet." over a table that has data. The branch is now split, with its own sentence and its own
  test. This is the one place the meaning change costs an extra branch instead of falling out free —
  the plan had claimed it cost none.
- **The plan sent the implementer into `_sentry.py`** (fourth pass), by saying "the three places
  that read it" when two of the three are in the file this issue may not touch. Corrected to name
  the `checks.py` reader only; the other two stay as note 2 for #488.
- **Three untested behaviours and three missed doc passages** (fourth pass): silencing with no
  roster (a missing `if roster_ids is not None` would turn the check permanently "Could not
  evaluate"), the resurrection cutoff boundary, and the wholly-silenced branch; plus
  `_MAX_LATE_SERIES_IN_TABLE`'s exactness claim, the `NOTE` above the `stale` filter, and
  `production-deployment.md`'s "green tick … yellow warning" sentence, which now has a third cause.
- **The rejection of `silenced_ids` minus `late` rested on a state that cannot arise** (fifth pass,
  from a parallel session): a series dropped from the roster. `upsert_metadata` never drops one.
  The conclusion survives by two other routes — there is no unfiltered `late` under this design, and
  a silenced id that was never in the roster would be misreported as resurrected — and the stated
  reason is corrected in both places it appeared. The same pass caught that PR #540 has already
  rewritten the `NOTE`, and that PR #544 — then open, since merged — reworks the docstring carrying
  the exactness claim, so both doc items now say what to look for rather than quoting text that has
  moved.
- **A departure from Jack's comment went unlisted** (fourth pass): his first comment sequences this
  issue after #439, #441 and #442, all still open. Now listed, with the reason it does not apply to
  this version.
- **The list is a `Final` tuple, not a YAML file** (third pass). The reachability attack did the
  work: the file would be git-tracked, baked into the image by `COPY conf/`, read on Fargate where
  nobody edits it, and already parsed by a CI test — so the parse guard defended a state reachable
  only by first deleting the test written to prevent it. This removed the `Settings` field, the
  `contracts` change, the loader, the guard, the degradation path, two tests and two doc
  amendments. Recorded as open question 1 because it overrules a comment of Jack's.
- **Filter `coverage` and `roster_ids`, not `late`** (third pass). Three lines at the top of the
  evaluator, and every count, description and Sentry field downstream becomes about the watched
  population automatically. This deleted the `n_silenced_late` field, its metadata key, the
  `All {n} watched` arithmetic and the test that pinned it. The price is that `n_series_total`
  changes meaning; nothing depends on the old one, and the new one is what makes the numbers true.
- **Two result fields, not three; the new parameter is keyword-only** (third pass), per the
  keyword-argument rule that landed in `code-style.md` with the merge.
- **Tests cut from ten to five plus one assertion** (third pass), after the earlier passes had
  already merged eleven into ten. The malformed-list and shipped-file tests go with the YAML.
- **No pydantic model, no id-to-reason mapping, no silenced-series metadata table, no second
  `_late_table_metadata` caller, no new cap constant** (first pass).
- **The healthy description was untrue** — `All 32 time series are up to date` when 30 were (second
  pass). Superseded by the third pass, which removed the arithmetic rather than correcting it.
- **The loader could mis-parse rather than degrade** (second pass): `"2333"` silences four series,
  `{23: null}` silences one by accident, `23.9` truncates. This finding is what made the third
  pass's case against the YAML concrete.
- **Two tests would not have failed on `main`** (second pass): the malformed-list test (a check with
  no reader also "does not raise and silences nothing") and the Sentry test, which extended a test
  that patches `evaluate_power_freshness` with a sentinel and would have asserted a property of the
  sentinel.
- **Four untested behaviours** (second pass): an id that exists nowhere, `is_healthy` staying `True`
  through a resurrection, the silencing clause on both description branches, and the metadata keys
  being emitted when nothing is silenced. All survive into the five tests above.
- **`operations.md` says `n_stale` and `n_never_reported` are "always exact"** (second pass) — added
  to the docs list as a rewrite.
- **Three `PowerFreshnessResult` construction sites** would break without defaults on the new fields
  (second pass). All three pass every field by keyword, so defaults fix all three with no edit.
- **Wording and omissions** (second pass): the Verdict claimed a resurrection "alerts" us, when with
  `_sentry.py` off limits it reports in Dagster's Checks view only; and Jack's suggestion of separate
  `known late` and `known missing` lists was dropped without being listed as a departure.
- **#523 collides textually** with the same function (second pass) — now open question 4.

### Rejected

- **"Drop the runtime fallback; let a malformed list hit the catch-all"** (first pass). Rejected at
  the time, on the grounds that an optional convenience must not blind the check's primary signal.
  The third pass reached the same destination by a better route: with no file there is no parse, so
  neither the fallback nor the catch-all route is needed.
- **"Make the path a module constant instead of a `Settings` field"** (first pass). Rejected at the
  time because the env var is how the existing `env` fixture redirects paths. The third pass showed
  the premise was wrong — `tests/test_checks.py` already monkeypatches attributes on `checks` five
  times — and the finding is accepted in its stronger form: no path at all.
- **"Compute resurrection as `silenced_ids` minus `late`"** (first pass). Still rejected, but the
  original reason was wrong and is corrected above. It said the subtraction would misreport "an id
  dropped from the roster with no rows on disk" — that state cannot arise. `upsert_metadata` merges
  each snapshot into the stored roster on `time_series_id` and never drops a series, so the roster
  holds every series we have ever seen and a series NGED stops publishing keeps its last stored
  values indefinitely (documented in PR #540, which rewrote the `NOTE` above the `stale` filter for
  the same reason). The rejection survives by two other routes: under this design there is no
  unfiltered `late` left to subtract from, and even in the structure where there is, a silenced id
  that was *never* in the roster — a typo in the constant, or an id NGED has never published — is
  absent from `late` and would be called resurrected. `test_silenced_series_are_withheld_and_others_still_warn`
  pins that case with id 404.
- **"Drop the `n_silenced` metadata key; the description says it in words"** (third pass, marginal).
  Rejected: every listing in this module is emitted beside a count (`n_late_listed`,
  `n_time_series_missing_listed`), and a string of ids is the one thing Dagster cannot plot or sort.
  One line.
- **"Cut the healthy-branch description test; after the refactor there is no arithmetic to get
  wrong"** (third pass). Rejected: requirement (a) is that we are told which ids are ignored
  *while everything looks fine*, and a clause appended only to the unhealthy branch would satisfy
  every other test. Taken in a cheaper form — the `_to_asset_check_result` test is parametrised over
  both branches rather than being two tests.
- **"Auto-silence anything stale beyond N days, with no list"** (third pass, weighed and rejected by
  the reviewer itself). It cannot satisfy requirement (b) without remembered state, and it silences
  a broken feed with nobody acknowledging it.
- **"'Nothing added can raise at all' is overstated"** (fourth pass) — accepted as a wording fix,
  not as a design change: a non-`int` entry would make `is_in` raise, but `Final[tuple[int, ...]]`
  plus `ty` in pre-commit and CI is the gate, and the catch-all still holds behind it. The sentence
  now says that.

### Fourth pass — confirmed sound

The reviewer walked filtering-the-inputs against filtering-`late` through every case — silenced and
stale, silenced and never-reported, silenced and fresh, silenced and present nowhere — and found
`late`, `n_stale` and `n_never` identical in each, with `n_series_total` the sole deliberate
divergence. It also confirmed on the installed Polars that an emptied `coverage` still yields "every
roster id is never", that the `last_time` dtype lookup is unaffected, and that the ordering matters
and the plan has it right: `resurrected_ids` is computed before the `coverage` filter.

(It walked a fifth case too — silenced, present in `coverage`, absent from the roster — which the
fifth pass then established cannot arise, because `upsert_metadata` never drops a series. Nothing
in the plan depends on it either way.)
