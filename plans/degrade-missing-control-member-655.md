# Degrade instead of raise on missing NWP control member (#655)

**Status: plan complete, three adversarial reviews done and folded in, awaiting human approval.**
No code has been written. Branch `degrade-missing-control-member-655` has had `main` merged in
(post-merge line numbers reflected throughout below). The `implement-issue` skill's step 1
(worktree setup) already ran for this plan, so a session resuming this work should check out this
branch into a new worktree (or reuse one if still present) and resume `implement-issue` at its
step 2 (implement, following the plan below), rather than re-running `plan-issue`. Size is now
**medium** (revised down from complex — see "Verdict, size, departures"), so one diff-correctness
review after implementation is enough; the plan needs nothing further from a human beyond a
decision on the three items in "Risks and open questions" below — everything else here is settled
and ready to implement as written.

**Problem.** `_engineer_features` (in `tabular_feature_engineer.py:216-224`) raises whenever
weather-lag features are requested and the NWP frame it was handed has no `ensemble_member == 0`
(control member) rows. This is **latent today, not live**: the guard only fires when
`weather_lags` (lags whose `base_col` isn't `"power"`) is non-empty, and the champion config
(`conf/model/xgboost.yaml`) selects no weather-lag features — only plain weather features and
power lags — a fact both `production_assets.py:289` and `ml_core/features/_lags.py:121` already
state in their own comments. So `live_forecasts` does not hit this raise on any slot today. It
would start hitting it the moment a future model config adds a weather-lag feature, at which point
a partial or malformed ECMWF ENS download that drops the control member would abort the whole slot
for every series and every ensemble member — a hard failure for what is, per
[inherent-stability rule 1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules),
the outside world misbehaving, not our bug. The state itself is reachable by design, not merely a
hypothetical: nothing between ingest and this join enforces the control member's presence
(`Nwp.ensemble_member` has no required-value constraint, `assess_nwp_run_completeness` is a
non-blocking `WARN` check, and `write_nwp` writes unconditionally regardless of completeness — see
`docs/architecture/ecmwf-ens-known-issues.md`, "Why it warns instead of failing the run"). No
recorded incident has hit this specific case; the two real NWP incidents on record
(`docs/live_service/intervention-log.md`, `docs/architecture/ecmwf-ens-known-issues.md`) are a
wholly-missing variable and a 50/51-member step dropout, both a different, fatal/malformed shape
that this guard doesn't gate.

**Solution.** Gate the raise on `power_fcst_init_time is None` — i.e. keep it in bulk
training/backtesting mode (the only mode any current caller uses to get there: both `cv_assets.py`
call sites), and drop it in single-run mode (production inference and replay backfills). With the
raise gone, a control-member-absent run flows into the weather-lag pipeline's own existing
join-miss path: `select_analysis_proxy` filters to `ensemble_member == 0`, finds nothing, and
`historical_weather` comes back empty, so every past-target-time weather lag left-joins to null —
structurally the same path already exercised, and tested, when the selected run is too fresh to
pass the publication-delay cut (`test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh`).
No new nulling logic is needed.

## Verdict, size, departures

**Worth implementing, with the scope narrowed.** The issue's premise checks out against the
mechanism (`tabular_feature_engineer.py:216-224`, `production_assets.py:328`) but not against its
claimed severity: the raise is latent, gated behind a feature the champion config never selects,
not something `live_forecasts` hits today (see Problem above). The fix it asks for — delete/scope
the raise, confirm the existing null-join path covers it, keep fail-fast for CV/training — is still
the right shape, with one addition the issue also asked for and the original plan wrongly declined
in full: the raise being removed was added deliberately, in commit `a330567f`, specifically to stop
a silent all-null weather-lag failure ("Fail loudly when no NWP control member for weather lag
features"). Removing it without adding a degradation record would reinstate exactly the silent
behaviour that commit fixed, so this plan now adds a log warning naming the run at fault (see "What
changes" below) to land as "degrade and record" rather than "degrade silently".

**Size: medium**, revised down from the issue's "complex". The premise that justified "complex" —
that `live_forecasts` hits this every slot — is false, and the actual change is a three-line mode
gate, one warning log call, and two tests, on a path no promoted model reaches today. Plan review
already ran (see below); one further diff-correctness review before the PR is enough.

**Departures from the issue body:**

- The issue asks to "confirm degradation actually reaches all three of the required channels —
  in-band, the warning table, and Sentry — not just doesn't raise." I'm not adding new machinery
  for any of them. In-band widening doesn't exist anywhere yet — today's model is a point forecast
  (`XGBoostConfig.objective` defaults to `reg:squarederror`), and band-widening needs quantile
  output ([#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263)) plus
  conformal calibration, neither shipped; the nulled feature still gets XGBoost's default-direction
  routing, the same substitute every other null feature in this pipeline relies on today. The
  warning table, `power_forecast_warnings`, is not yet in code at all
  (`docs/roadmap/delivery-tables.md`: "🚧 Planned ... not yet in code"), so there is nothing to
  hook into. Sentry is already covered one layer upstream: `ecmwf_ens`'s
  `assess_nwp_run_completeness` (`assets.py:438,519`) turns a missing control member into a WARN
  `AssetCheckResult` naming `missing_ensemble_members` in Dagster's Checks view at ingest — the same
  "Checks view only, no proactive Sentry push on a WARN pass" level every other check in
  `checks.py`/`assets.py` is at today, a gap tracked under
  [#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501) rather than specific
  to this failure mode. Closing #501 generally, or building a first slice of
  `power_forecast_warnings`, is materially bigger than this bug fix; see Risks below.

## What changes, file by file

**`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py`**

- `_engineer_features`: wrap the existing `if nwp_lf is not None and weather_lags and
  nwp_lf.filter(...).limit(1).collect().is_empty(): raise ValueError(...)` block (currently at
  lines 216-224) in an additional `power_fcst_init_time is None` condition, so it only fires in
  bulk mode. Single-run mode (used by both live inference and replay backfills) skips the check
  entirely and falls through to the existing `historical_weather` construction and
  `_apply_weather_lag` join, which already produces nulls when `historical_weather` is empty.
- In the single-run branch, when that same emptiness condition holds (weather lags requested, no
  control member present), log a warning naming `nwp_init_time` and the affected series before
  falling through — e.g. `logger.warning("NWP run %s has no control member (ensemble_member == 0); "
  "weather lag features will be null for this slot", nwp_init_time)`. This is the degradation
  record the issue asked for and the original raise (`a330567f`, "Fail loudly when no NWP control
  member for weather lag features") existed to guarantee some form of; without it, removing the
  raise would make the failure silent rather than recorded, which the previous version of this plan
  wrongly accepted as the whole fix. This is not a new detection mechanism — it does not touch
  `assess_nwp_run_completeness` or Sentry — it is the minimum needed so the failure leaves a trace
  in the `live_forecasts` run logs.
- Update the `ValueError` message to make clear it's a bulk-mode/training guard (e.g. "... to build
  historical weather during bulk training or backtesting, but no such rows were found ..."), so a
  future reader hitting it in a traceback isn't confused about why single-run mode never sees it.
- Module docstring (lines 12-18, "Lazy Evaluation"): rewrite the control-member-check sentence
  (currently lines 16-18) to say the eager `.collect()` guard runs only in bulk mode
  (training/backtesting fail-fast); in single-run mode (production, replay) a missing control
  member is absent input, not a contract violation, and degrades weather lags to null through the
  ordinary join-miss path, with a warning logged naming the run.

**`src/nged_substation_forecast/defs/production_assets.py`**

- `live_forecasts` (function now at line 241, `engineer()` call at line 328) docstring (the
  existing paragraph starting "Note: only one NWP run is loaded here...", now around line 289):
  extend it to also cover this case — a control-member-absent run nulls every weather lag feature
  for that slot and logs a warning naming the run, the same way a too-fresh run already does,
  rather than failing the slot. Point at the new test (see below) alongside the existing
  `test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh` reference. Also correct the
  existing "None are in the current champion config" sentence here to note this is exactly the
  condition that keeps today's raise latent rather than live.

**`docs/architecture/performance.md`**

- Line 61 ("The one exception is a `limit(1).collect()` guard...fails loudly instead of silently
  returning an empty frame" — text unchanged by the `main` merge, only its line number moved from
  58 to 61): qualify to say this applies in bulk/training mode only, so the "always eager, always
  fails loudly" framing doesn't contradict the new single-run behaviour, and add a clause noting
  single-run mode instead logs a warning rather than failing silently.

## Design-philosophy check

This is squarely the production-degradation path inherent-stability.md describes. A missing
control member is the outside world misbehaving (a partial/malformed ECMWF ENS download), not our
bug, so per rule 1 it must degrade rather than raise — and per rule 4, the degradation must be
recorded, not silent, which is why this revision of the plan adds the warning log above. The fix
keeps the CV/training raise (rule 9: R&D fails fast, because a silently-degraded training run
poisons every comparison built on it, while production fails forward). No asset check changes, so
rule 6/7 (WARN, non-blocking, guarded-body) aren't touched — `assess_nwp_run_completeness` (now in
`packages/contracts/src/contracts/weather_schemas.py:761-834`, called from `assets.py:491`)'s
existing check already covers detection at ingest time; this plan's warning log covers the
separate, later point where a bad run actually affects a forecast. Delivers toward
[H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself):
one fewer way for `live_forecasts` to need a human to intervene on a 6-hourly cadence.

## Tests

**`packages/ml_core/tests/test_features.py`**

- Rename and rework `test_engineer_features_raises_when_no_control_member_for_weather_lag`
  (currently at lines 1077-1103; its name asserts a raise, which is no longer true in single-run
  mode) to assert the *new* behaviour: no raise, the weather-lag column comes back entirely null,
  and the warning is logged (via `caplog` or equivalent). **Fix the fixture's timing, not just the
  assertion**: today's fixture sets `valid_time` 12:00 with a `temperature_2m_lag_6h` feature and
  `power_fcst_init_time` 06:00, so `target_time` (`valid_time - lag_hours`) lands exactly equal to
  `power_fcst_init_time`, which `_apply_weather_lag`'s `>=` boundary (`_lags.py:123`) routes to the
  *same-run* join, not the freshest-run join that consumes `historical_weather` — so the null it
  observes today is really "no NWP row at that instant", not "no control member". Give the
  reworked test a `target_time` strictly *before* `power_fcst_init_time`, and keep
  `nwp_init_time + nwp_publication_delay_hours <= power_fcst_init_time` (the default is 9h) so
  `select_analysis_proxy`'s `available_at` cut doesn't filter the run out for an unrelated
  freshness reason — that isolates the null to the one cause this test exists to prove: control
  member absent, `select_analysis_proxy` filters to `ensemble_member == 0` and returns nothing, so
  the freshest-run join misses. This is the test that would fail on `main` today (it currently
  asserts a raise) and, once corrected as above, is the one that actually pins the mechanism this
  plan's "Solution" section describes.
- Add `test_engineer_features_raises_in_bulk_mode_when_no_control_member_for_weather_lag`: same
  fixture shape (NWP with only `ensemble_member=1`, a weather-lag feature requested), called with
  `power_fcst_init_time=None` (bulk mode). Asserts the `ValueError` still raises. This is new
  coverage — nothing today distinguishes bulk-mode from single-run-mode for this check, so this
  test is what would catch a future change accidentally dropping the raise everywhere. The bulk-mode
  check runs before any join, so this test's fixture timing doesn't need the same care as the one
  above.

**`tests/test_live_forecasts.py`**

- Add a slim integration test alongside `test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh`:
  reuse `_save_model_trained_on_weather_lag` to train on a weather lag, write NWP records for the
  selected run with only non-control members (e.g. `ensemble_member=1`, no `0`), materialise
  `live_forecasts`, and assert `result.success` — no predict-spy or captured-frame scaffolding,
  since the corrected unit test above already pins *which* column goes null and by what mechanism,
  with no layer between it and `live_forecasts`' own call (`production_assets.py:316` calls `engineer()`
  directly, nothing wraps the raise). This test's job is narrower: confirm the materialisation
  itself doesn't fail — the fact that would be false on `main` today. It is also a tripwire for the
  fan-out gap in "Risks and open questions" below: if that gap ever turned into a real crash rather
  than a silently-missing row, this is the test that would catch it.

## Docs to update

- `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py` module docstring (as above).
- `docs/architecture/performance.md` (as above).
- `src/nged_substation_forecast/defs/production_assets.py`'s `live_forecasts` docstring (as above).
- No roadmap "Implementation details" section covers this issue and no milestone status banner
  changes — #655 isn't a roadmap item, it's a bug fix.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest packages/ml_core/tests/test_features.py
uv run pytest tests/test_live_forecasts.py
uv run pytest                      # full suite, green-before-push
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

## Risks and open questions

1. **A missing control member also drops that member's own forecast row, not just its
   weather-lag history — this issue doesn't reach that.** `_join_nwp_single_run`
   (`_nwp.py:79-80`) left-joins power onto NWP on `(time_series_id, valid_time, nwp_init_time)`,
   *not* `ensemble_member`, so the row for `ensemble_member == 0` fans in from whichever NWP rows
   match — if the control member is wholly absent from the run, no row for it is produced at all
   (not null: absent). That's a second, separate failure surface from the weather-lag nulling
   `_apply_weather_lag` handles, and this plan doesn't touch it: the issue is scoped to the raise
   in `_engineer_features`, and this row-count gap is a pre-existing property of the single-run
   join, unrelated to the raise being removed. Recommend a follow-up issue rather than folding it
   in here.
2. **Should this issue also add a proactive Sentry event for a WARN-passing completeness check?**
   I've argued no (see Departures above) — it's a cross-cutting gap ([#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501))
   affecting every WARN check in the codebase, not specific to this failure mode, and bolting one
   Sentry call onto `assess_nwp_run_completeness` alone would be an inconsistent, one-off exception
   to how every other check in `checks.py`/`assets.py` behaves today. My recommendation is to leave
   it to #501 and note the connection there. A human reviewer who wants #655 to close that gap for
   this one check specifically should say so.
3. **Does "single-run mode" ever legitimately want the old fail-fast behaviour?** Single-run mode
   also covers replay backfills (`select_nwp_init_time`'s `"replay"` availability mode). A backfill
   of a slot with a genuinely broken NWP run would now silently null its weather lags rather than
   raising. I think that's correct — a backfill re-processing the past should behave like the live
   slot it's replaying, not diverge — but flagging it since the issue text focuses on the *live*
   call site specifically.
4. **Naming:** the issue calls the guard "the NWP control-member check" throughout; I've kept that
   name in the updated docstring/error message rather than inventing new vocabulary, since it's
   already used in `docs/architecture/performance.md` and the tests.

## First review (simplicity)

A fresh sub-agent verified the core mechanism against the code (no crash path in
`select_analysis_proxy`, `_apply_weather_lag`, `_upsample_nwp_to_half_hourly` or
`_select_output_columns` when the control member is absent), confirmed the gate condition is
already the simplest signal available (both CV call sites omit `power_fcst_init_time`, i.e. call in
bulk mode), and found the scope cut on in-band/warning-table/Sentry machinery legitimate. Two
findings taken into the plan above: the "Departures" section was cut from three sub-bulleted
justifications to one paragraph per the prose-style rule, and the new integration test was slimmed
to drop redundant predict-spy/frame-capture scaffolding, keeping only the `result.success` check as
a tripwire. One new risk taken in: the fan-out gap in `_join_nwp_single_run` (risk 1 above) — a
real, verified, separate failure surface this issue doesn't touch. No findings rejected.

## Second review (correctness and testability)

A second fresh sub-agent independently confirmed the plan's account of `main` (the raise fires
unconditionally today, both CV call sites are bulk mode, `live_forecasts` is single-run mode),
found the fix itself fails closed nowhere (`_select_output_columns` checks column names not
nullability; `XGBoostForecaster` casts null to `NaN`, which XGBoost's default-direction routing
already handles), and confirmed no missed caller, doc or schema. One real finding taken into the
plan above: the retooled unit test's original fixture set `target_time` exactly equal to
`power_fcst_init_time`, which routes to the *same-run* join rather than the freshest-run join that
actually depends on the control member — so, as first written, that test would have passed for the
wrong reason (no NWP row at that instant, not "no control member"). The Tests section above now
specifies the timing fix. Also taken in: renaming the retooled test, since its old name asserts a
raise that no longer happens. One finding noted but not actioned: the plan doesn't cross-reference
`inherent-stability.md`'s separate mention of the *different*, already-tracked raise in
`select_nwp_init_time` (issue #446) — correctly out of scope for #655, and the reviewer agreed it's
cosmetic rather than a defect, so left as is.

## Third review (necessity, and staleness after the `main` merge)

Run after `main` was merged into this branch (~160 files changed upstream since this plan was
written), with two questions: is this fix needed at all, and is the plan still accurate against the
merged code. On necessity: **implement, with scope reduced**, not "don't implement" and not
"unchanged". The raise cannot fire in production today — it is gated behind `weather_lags` being
non-empty, and the champion config selects none — so the Problem statement's "hits every 6-hourly
slot" claim was false, both in this plan and in the issue body it was copied from; size dropped from
complex to medium accordingly. The state itself is reachable by design (ingest has no control-member
completeness gate stronger than a non-blocking WARN check), so the fix is still worth making, just
not urgently and not at complex-review cost. The review also found the raise being deleted was added
deliberately in commit `a330567f` to stop a silent all-null failure, and that the plan's original
"Departures" section declined all three of the issue's requested degradation channels without
proposing any substitute record — reinstating exactly the silent behaviour `a330567f` fixed. This
plan now adds a warning log naming the run at fault as the minimum fix for that gap (see "What
changes" above). On staleness: the mechanism the plan depends on is fully intact after the merge —
no function was renamed, no test was removed, no CV call site's mode changed — only line numbers
moved (this plan's line references above are updated accordingly), and `assess_nwp_run_completeness`
moved from `assets.py` into `packages/contracts/src/contracts/weather_schemas.py:761-834`. Net: the
plan needed content edits (Problem statement, size, the added warning log) rather than a re-plan
from scratch; both are folded in above.
